"""
Regularized LoRA Fine-Tuning Training Script (Exp12)
====================================================
Strong regularization (MixUp, CutMix, label smoothing) + hard attention
masking for V-JEPA CVS classification.

Key features:
  - All Exp10b LoRA features (frozen backbone + LoRA adapters + task heads)
  - MixUp / CutMix batch augmentation on pixel values
  - Label smoothing on CVS binary labels
  - Hard attention masking: zero out background spatial tokens using seg masks
"""

import argparse
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_multitask import MultiTaskCVSDataset, collate_fn
from dataset_combined_multitask import CombinedMultiTaskDataset, collate_fn as combined_collate_fn
from utils import (
    AverageMeter,
    EarlyStopping,
    compute_metrics,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logging,
)

# Import peft for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: peft not installed. Run: pip install peft")


# ============================================================================
# New augmentation / regularization functions
# ============================================================================

def mixup_data(videos: torch.Tensor, labels: torch.Tensor, alpha: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    MixUp augmentation: blend video pairs and their labels using Beta distribution.

    Args:
        videos: [B, T, C, H, W] pixel values
        labels: [B, num_classes] binary labels
        alpha: Beta distribution parameter

    Returns:
        mixed_videos, mixed_labels, lam
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = videos.size(0)
    index = torch.randperm(batch_size, device=videos.device)

    mixed_videos = lam * videos + (1 - lam) * videos[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_videos, mixed_labels, lam


def cutmix_data(videos: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    CutMix augmentation: paste random rectangle from shuffled sample onto each video.
    Same rectangle applied across all T frames.

    Args:
        videos: [B, T, C, H, W] pixel values
        labels: [B, num_classes] binary labels
        alpha: Beta distribution parameter

    Returns:
        mixed_videos, mixed_labels, lam
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = videos.size(0)
    index = torch.randperm(batch_size, device=videos.device)

    _, T, C, H, W = videos.shape

    # Compute cut rectangle
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # Random center
    cy = np.random.randint(H)
    cx = np.random.randint(W)

    # Clamp to image bounds
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    # Paste rectangle from shuffled sample
    mixed_videos = videos.clone()
    mixed_videos[:, :, :, y1:y2, x1:x2] = videos[index, :, :, y1:y2, x1:x2]

    # Adjust lambda to actual cut area
    lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (H * W)

    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_videos, mixed_labels, lam


def apply_batch_augmentation(
    videos: torch.Tensor,
    labels: torch.Tensor,
    aug_config: dict,
) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
    """
    Apply MixUp OR CutMix OR nothing (mutually exclusive).

    Args:
        videos: [B, T, C, H, W] pixel values
        labels: [B, num_classes] labels
        aug_config: dict with mixup_alpha, cutmix_alpha, mixup_prob, cutmix_prob

    Returns:
        (videos, labels, lam, augmented_flag)
    """
    mixup_prob = aug_config.get("mixup_prob", 0.0)
    cutmix_prob = aug_config.get("cutmix_prob", 0.0)
    mixup_alpha = aug_config.get("mixup_alpha", 0.8)
    cutmix_alpha = aug_config.get("cutmix_alpha", 1.0)

    r = np.random.rand()

    if r < mixup_prob:
        videos, labels, lam = mixup_data(videos, labels, mixup_alpha)
        return videos, labels, lam, True
    elif r < mixup_prob + cutmix_prob:
        videos, labels, lam = cutmix_data(videos, labels, cutmix_alpha)
        return videos, labels, lam, True
    else:
        return videos, labels, 1.0, False


def smooth_labels(labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    Apply label smoothing to binary labels.
    Transforms: 0 -> smoothing/2, 1 -> 1 - smoothing/2

    Args:
        labels: [B, num_classes] binary labels (0 or 1)
        smoothing: smoothing factor

    Returns:
        Smoothed labels
    """
    return labels * (1.0 - smoothing) + (1.0 - labels) * smoothing


def create_spatial_mask_from_segmentation(
    seg_masks: torch.Tensor,
    mask_batch_indices: torch.Tensor,
    batch_size: int,
    spatial_size: int = 16,
    num_temporal_bins: int = 8,
) -> torch.Tensor:
    """
    Create [B, num_tokens] float mask from segmentation annotations.
    Anatomy (class > 0) -> 1.0, background (class == 0) -> 0.0.
    Tiles spatial mask across temporal bins.
    Batch elements without masks default to all-ones (no masking).

    Args:
        seg_masks: [N, H_mask, W_mask] segmentation masks (class indices)
        mask_batch_indices: [N] which batch element each mask belongs to
        batch_size: total batch size B
        spatial_size: spatial grid size (e.g. 16 for 16x16 tokens)
        num_temporal_bins: number of temporal bins in token sequence

    Returns:
        [B, num_tokens] float mask where num_tokens = num_temporal_bins * spatial_size^2
    """
    spatial_tokens = spatial_size * spatial_size
    num_tokens = num_temporal_bins * spatial_tokens
    device = seg_masks.device

    # Default: all ones (no masking)
    token_mask = torch.ones(batch_size, num_tokens, device=device)

    if len(mask_batch_indices) == 0:
        return token_mask

    # Get unique batch indices that have masks
    unique_batch_ids = torch.unique(mask_batch_indices)

    for b_idx in unique_batch_ids:
        b = b_idx.item()
        # Get all masks for this batch element
        mask_selector = (mask_batch_indices == b_idx)
        batch_masks = seg_masks[mask_selector]  # [K, H_mask, W_mask]

        # Use first mask as representative spatial mask
        # Downsample to spatial_size x spatial_size
        mask = batch_masks[0].float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        downsampled = F.interpolate(
            mask, size=(spatial_size, spatial_size), mode="nearest"
        ).squeeze()  # [spatial_size, spatial_size]

        # Anatomy (class > 0) -> 1.0, background -> 0.0
        spatial_mask = (downsampled > 0).float().view(-1)  # [spatial_tokens]

        # Tile across all temporal bins
        tiled_mask = spatial_mask.unsqueeze(0).expand(num_temporal_bins, -1).reshape(-1)  # [num_tokens]

        token_mask[b] = tiled_mask

    return token_mask


def apply_hard_attention_mask(
    features: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Element-wise multiply features by mask to zero out background tokens.

    Args:
        features: [B, num_tokens, D] feature tensor
        token_mask: [B, num_tokens] float mask (1.0 = keep, 0.0 = zero out)

    Returns:
        Masked features [B, num_tokens, D]
    """
    return features * token_mask.unsqueeze(-1)


# ============================================================================
# Model classes (from train_lora.py)
# ============================================================================

class AttentionPooling(nn.Module):
    """Learnable attention pooling over token dimension."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        attn_out, _ = self.attention(query, x, x)
        return self.norm(attn_out.squeeze(1))


class LightweightSegDecoder(nn.Module):
    """Lightweight segmentation decoder for V-JEPA features."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_classes: int = 5,
        output_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.output_size = output_size

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        frame_indices: torch.Tensor,
        batch_indices: torch.Tensor,
        num_temporal_bins: int = 8,
        spatial_size: int = 16,
    ) -> torch.Tensor:
        if len(frame_indices) == 0:
            return torch.zeros(0, self.num_classes, self.output_size, self.output_size,
                             device=features.device)

        B, num_tokens, D = features.shape
        spatial_tokens = spatial_size * spatial_size

        temporal_indices = frame_indices // 2

        frame_features = []
        for i, (batch_idx, temp_idx) in enumerate(zip(batch_indices, temporal_indices)):
            start_idx = temp_idx * spatial_tokens
            end_idx = start_idx + spatial_tokens

            if end_idx > num_tokens:
                end_idx = num_tokens
                start_idx = max(0, end_idx - spatial_tokens)

            spatial_feats = features[batch_idx, start_idx:end_idx]

            if spatial_feats.shape[0] < spatial_tokens:
                padding = torch.zeros(spatial_tokens - spatial_feats.shape[0], D,
                                     device=features.device)
                spatial_feats = torch.cat([spatial_feats, padding], dim=0)

            frame_features.append(spatial_feats)

        frame_features = torch.stack(frame_features, dim=0)
        frame_features = self.proj(frame_features)

        N = frame_features.shape[0]
        frame_features = frame_features.view(N, spatial_size, spatial_size, -1)
        frame_features = frame_features.permute(0, 3, 1, 2).contiguous()

        logits = self.decoder(frame_features)

        return logits


class VJEPA_LoRA(nn.Module):
    """
    V-JEPA model with LoRA adapters for efficient fine-tuning.

    Architecture:
        V-JEPA Encoder (frozen) + LoRA Adapters (trainable)
            |
            +-> [Hard Attention Mask] -> Attention Pooling -> CVS Head -> 3 logits
            |
            +-> Seg Decoder -> per-frame segmentation masks
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
        hidden_dim: int = 1024,
        # LoRA params
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
        # CVS head params
        cvs_hidden: int = 512,
        cvs_dropout: float = 0.5,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        # Seg head params
        num_seg_classes: int = 5,
        seg_output_size: int = 64,
        seg_dropout: float = 0.1,
    ):
        super().__init__()

        if not PEFT_AVAILABLE:
            raise ImportError("peft library required. Install with: pip install peft")

        self.model_name = model_name
        self.hidden_dim = hidden_dim

        # Load V-JEPA backbone
        print(f"Loading V-JEPA model: {model_name}")
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(model_name)

        # Convert to float32 for stable training
        backbone = backbone.float()

        # Freeze all backbone parameters first
        for param in backbone.parameters():
            param.requires_grad = False

        # Find target modules for LoRA
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        # Find actual module names in the model
        actual_target_modules = self._find_target_modules(backbone, target_modules)
        print(f"Found {len(actual_target_modules)} modules to apply LoRA")

        if len(actual_target_modules) == 0:
            # Try alternative patterns
            print("Trying alternative module patterns...")
            actual_target_modules = self._find_target_modules_alternative(backbone)
            print(f"Found {len(actual_target_modules)} modules with alternative patterns")

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=actual_target_modules if actual_target_modules else target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=[],  # Don't save any additional modules
        )

        # Apply LoRA to backbone
        print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}")
        self.backbone = get_peft_model(backbone, lora_config)

        # Print trainable parameters
        self.backbone.print_trainable_parameters()

        # CVS classification head
        self.pooler = AttentionPooling(
            hidden_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )

        self.cvs_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, cvs_hidden),
            nn.GELU(),
            nn.Dropout(cvs_dropout),
            nn.Linear(cvs_hidden, 3),
        )

        # Segmentation head
        self.seg_head = LightweightSegDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_seg_classes,
            output_size=seg_output_size,
            dropout=seg_dropout,
        )

        self._init_heads()

    def _find_target_modules(self, model, target_patterns: List[str]) -> List[str]:
        """Find module names that match target patterns."""
        matched_modules = []

        for name, module in model.named_modules():
            for pattern in target_patterns:
                if pattern in name and isinstance(module, nn.Linear):
                    matched_modules.append(name)
                    break

        return list(set(matched_modules))

    def _find_target_modules_alternative(self, model) -> List[str]:
        """Try alternative patterns for finding attention projections."""
        matched_modules = []

        # Common patterns in ViT models
        patterns = [
            r'.*attention.*query.*',
            r'.*attention.*value.*',
            r'.*attn.*q_proj.*',
            r'.*attn.*v_proj.*',
            r'.*self_attn.*q_proj.*',
            r'.*self_attn.*v_proj.*',
            r'.*qkv.*',  # Some models use combined qkv
        ]

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for pattern in patterns:
                    if re.match(pattern, name, re.IGNORECASE):
                        matched_modules.append(name)
                        break

        # If still nothing, just target all attention-related linear layers
        if not matched_modules:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and 'attn' in name.lower():
                    matched_modules.append(name)

        return list(set(matched_modules))

    def _init_heads(self):
        """Initialize classification head weights."""
        for module in self.cvs_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        mask_frame_indices: torch.Tensor = None,
        mask_batch_indices: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass for both tasks.

        Args:
            pixel_values_videos: [B, T, C, H, W] input video
            mask_frame_indices: frame indices for segmentation
            mask_batch_indices: batch indices for segmentation
            attention_mask: [B, num_tokens] hard attention mask (optional).
                            1.0 = keep token, 0.0 = zero out (background).
        """
        # Get V-JEPA features through LoRA-adapted backbone
        features = self.backbone.get_vision_features(
            pixel_values_videos=pixel_values_videos
        )

        features = features.float()

        # Apply hard attention mask before pooler/heads
        if attention_mask is not None:
            features = apply_hard_attention_mask(features, attention_mask)

        # CVS classification
        pooled = self.pooler(features)
        cvs_logits = self.cvs_head(pooled)

        result = {"cvs_logits": cvs_logits}

        # Segmentation
        if mask_frame_indices is not None and mask_batch_indices is not None and len(mask_frame_indices) > 0:
            seg_logits = self.seg_head(
                features,
                mask_frame_indices,
                mask_batch_indices,
            )
            result["seg_logits"] = seg_logits

        return result

    def process_videos(self, videos, device: torch.device) -> torch.Tensor:
        """Process videos for V-JEPA input."""
        if isinstance(videos, torch.Tensor):
            video_tensor = videos
            if video_tensor.dim() == 5:
                B, T, dim3, H, W = video_tensor.shape
                if dim3 == 3 and video_tensor.dtype == torch.float32:
                    if video_tensor.min() < -0.5 or video_tensor.max() > 1.5:
                        return video_tensor.to(device)
                    else:
                        video_tensor = video_tensor.to(device)
                        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
                        return (video_tensor - mean) / std
                elif W == 3:
                    video_tensor = video_tensor.permute(0, 1, 4, 2, 3).contiguous()
                    video_tensor = video_tensor.to(device).float() / 255.0
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
                    return (video_tensor - mean) / std
            raise ValueError(f"Unexpected tensor shape: {video_tensor.shape}")

        elif isinstance(videos, list):
            if len(videos) == 0:
                raise ValueError("Empty video list")
            first = videos[0]
            if isinstance(first, np.ndarray):
                video_tensor = torch.from_numpy(np.stack(videos))
            elif isinstance(first, torch.Tensor):
                video_tensor = torch.stack(videos)
            else:
                raise ValueError(f"Unsupported video type: {type(first)}")

            video_tensor = video_tensor.permute(0, 1, 4, 2, 3).contiguous()
            video_tensor = video_tensor.to(device).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
            return (video_tensor - mean) / std

        raise ValueError(f"Unsupported videos type: {type(videos)}")

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_lora_params(self):
        """Get LoRA parameters for optimizer."""
        lora_params = []
        for name, param in self.backbone.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                lora_params.append(param)
        return lora_params

    def get_head_params(self):
        """Get head parameters for optimizer."""
        head_params = []
        for param in self.pooler.parameters():
            if param.requires_grad:
                head_params.append(param)
        for param in self.cvs_head.parameters():
            if param.requires_grad:
                head_params.append(param)
        for param in self.seg_head.parameters():
            if param.requires_grad:
                head_params.append(param)
        return head_params


class MultiTaskLoss(nn.Module):
    """Combined loss for CVS classification and segmentation."""

    def __init__(
        self,
        cvs_weight: float = 1.0,
        seg_weight: float = 0.3,
        cvs_pos_weight: list = None,
        seg_class_weights: list = None,
    ):
        super().__init__()
        self.cvs_weight = cvs_weight
        self.seg_weight = seg_weight

        if cvs_pos_weight is not None:
            pos_weight = torch.tensor(cvs_pos_weight, dtype=torch.float32)
            self.cvs_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.cvs_loss = nn.BCEWithLogitsLoss()

        if seg_class_weights is not None:
            weight = torch.tensor(seg_class_weights, dtype=torch.float32)
            self.seg_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        else:
            self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, cvs_logits, cvs_labels, seg_logits=None, seg_labels=None):
        if hasattr(self.cvs_loss, 'pos_weight') and self.cvs_loss.pos_weight is not None:
            self.cvs_loss.pos_weight = self.cvs_loss.pos_weight.to(cvs_logits.device)

        cvs_loss = self.cvs_loss(cvs_logits, cvs_labels)

        if seg_logits is not None and seg_labels is not None and seg_logits.shape[0] > 0:
            if hasattr(self.seg_loss, 'weight') and self.seg_loss.weight is not None:
                self.seg_loss.weight = self.seg_loss.weight.to(seg_logits.device)
            seg_loss = self.seg_loss(seg_logits, seg_labels)
            total_loss = self.cvs_weight * cvs_loss + self.seg_weight * seg_loss
        else:
            seg_loss = torch.tensor(0.0, device=cvs_logits.device)
            total_loss = self.cvs_weight * cvs_loss

        return {"total_loss": total_loss, "cvs_loss": cvs_loss, "seg_loss": seg_loss}


def compute_seg_metrics(seg_logits, seg_labels, num_classes=5):
    """Compute segmentation metrics."""
    if seg_logits.shape[0] == 0:
        return {"seg_miou": 0.0, "seg_acc": 0.0}

    preds = seg_logits.argmax(dim=1)
    preds_flat = preds.view(-1)
    labels_flat = seg_labels.view(-1)

    valid_mask = labels_flat != 255
    preds_flat = preds_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]

    if len(labels_flat) == 0:
        return {"seg_miou": 0.0, "seg_acc": 0.0}

    acc = (preds_flat == labels_flat).float().mean().item()

    ious = []
    for c in range(num_classes):
        pred_c = preds_flat == c
        label_c = labels_flat == c
        intersection = (pred_c & label_c).sum().float()
        union = (pred_c | label_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())

    miou = np.mean(ious) if ious else 0.0
    return {"seg_miou": miou, "seg_acc": acc}


def create_cosine_scheduler_with_warmup(
    optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.01,
):
    """Create cosine annealing scheduler with warmup."""
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, config, scaler, logger):
    """Train for one epoch with MixUp/CutMix, label smoothing, and hard attention masking."""
    model.train()

    loss_meter = AverageMeter()
    cvs_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    accum_steps = config.get("gradient_accumulation", 1)
    use_amp = config.get("mixed_precision", True)

    # Augmentation and masking config
    aug_config = config.get("augmentation", {})
    ham_config = config.get("hard_attention_masking", {})
    label_smoothing = config.get("label_smoothing", 0.0)
    ham_enabled = ham_config.get("enabled", False) and ham_config.get("apply_during_training", False)
    ham_spatial_size = ham_config.get("spatial_size", 16)

    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        videos = batch["videos"]
        cvs_labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)
        mask_frame_indices = batch["mask_frame_indices"].to(device)
        mask_batch_indices = batch["mask_batch_indices"].to(device)

        pixel_values = model.process_videos(videos, device)

        # Save original labels for metric collection (before augmentation/smoothing)
        original_cvs_labels = cvs_labels.clone()

        # Apply MixUp/CutMix to pixel_values and cvs_labels (NOT seg masks)
        pixel_values, cvs_labels, lam, augmented = apply_batch_augmentation(
            pixel_values, cvs_labels, aug_config
        )

        # Apply label smoothing to CVS labels
        if label_smoothing > 0:
            cvs_labels = smooth_labels(cvs_labels, label_smoothing)

        # Create hard attention mask from segmentation masks (when available)
        attention_mask = None
        if ham_enabled and len(mask_batch_indices) > 0:
            # Infer num_temporal_bins from feature dimensions
            # V-JEPA ViT-L with 16 frames, fpc=16: num_tokens = 8 * 16 * 16 = 2048
            num_temporal_bins = 8  # 16 frames / 2 (temporal stride)
            attention_mask = create_spatial_mask_from_segmentation(
                masks, mask_batch_indices,
                batch_size=pixel_values.size(0),
                spatial_size=ham_spatial_size,
                num_temporal_bins=num_temporal_bins,
            )

        is_accumulating = ((batch_idx + 1) % accum_steps != 0) and (batch_idx + 1 < len(train_loader))

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(pixel_values, mask_frame_indices, mask_batch_indices, attention_mask)
                seg_logits = outputs.get("seg_logits", None)
                loss_dict = criterion(
                    outputs["cvs_logits"], cvs_labels,
                    seg_logits, masks if seg_logits is not None else None
                )
                loss = loss_dict["total_loss"] / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()

            if not is_accumulating:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            outputs = model(pixel_values, mask_frame_indices, mask_batch_indices, attention_mask)
            seg_logits = outputs.get("seg_logits", None)
            loss_dict = criterion(
                outputs["cvs_logits"], cvs_labels,
                seg_logits, masks if seg_logits is not None else None
            )
            loss = loss_dict["total_loss"] / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            if not is_accumulating:
                nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        loss_meter.update(loss_dict["total_loss"].item(), original_cvs_labels.size(0))
        cvs_loss_meter.update(loss_dict["cvs_loss"].item(), original_cvs_labels.size(0))
        if seg_logits is not None and seg_logits.shape[0] > 0:
            seg_loss_meter.update(loss_dict["seg_loss"].item(), seg_logits.size(0))

        # Use original (unsmoothed, unmixed) labels for metric collection
        with torch.no_grad():
            probs = torch.sigmoid(outputs["cvs_logits"])
            all_preds.append(probs.cpu().numpy())
            all_targets.append(original_cvs_labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "cvs": f"{cvs_loss_meter.avg:.4f}"})

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    cvs_metrics = compute_metrics(all_preds, all_targets, 0.5)

    return {
        "loss": loss_meter.avg,
        "cvs_loss": cvs_loss_meter.avg,
        "seg_loss": seg_loss_meter.avg,
        "mAP": cvs_metrics["mAP"],
        "AP_C1": cvs_metrics.get("AP_C1", 0),
        "AP_C2": cvs_metrics.get("AP_C2", 0),
        "AP_C3": cvs_metrics.get("AP_C3", 0),
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    """Validate the model. No MixUp/CutMix, no label smoothing, no hard attention masking."""
    model.eval()

    loss_meter = AverageMeter()
    all_preds = []
    all_targets = []
    seg_miou_sum = 0.0
    seg_count = 0

    for batch in tqdm(val_loader, desc="Validating"):
        videos = batch["videos"]
        cvs_labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)
        mask_frame_indices = batch["mask_frame_indices"].to(device)
        mask_batch_indices = batch["mask_batch_indices"].to(device)

        pixel_values = model.process_videos(videos, device)
        outputs = model(pixel_values, mask_frame_indices, mask_batch_indices)

        seg_logits = outputs.get("seg_logits", None)
        loss_dict = criterion(
            outputs["cvs_logits"], cvs_labels,
            seg_logits, masks if seg_logits is not None else None
        )

        loss_meter.update(loss_dict["total_loss"].item(), cvs_labels.size(0))

        if seg_logits is not None and seg_logits.shape[0] > 0:
            seg_metrics = compute_seg_metrics(seg_logits, masks)
            seg_miou_sum += seg_metrics["seg_miou"] * seg_logits.size(0)
            seg_count += seg_logits.size(0)

        probs = torch.sigmoid(outputs["cvs_logits"])
        all_preds.append(probs.cpu().numpy())
        all_targets.append(cvs_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    cvs_metrics = compute_metrics(all_preds, all_targets, 0.5)

    return {
        "loss": loss_meter.avg,
        "mAP": cvs_metrics["mAP"],
        "AP_C1": cvs_metrics.get("AP_C1", 0),
        "AP_C2": cvs_metrics.get("AP_C2", 0),
        "AP_C3": cvs_metrics.get("AP_C3", 0),
        "seg_miou": seg_miou_sum / max(seg_count, 1),
    }


# Import augmented dataset from train_staged
try:
    from train_staged import AugmentedMultiTaskDataset
except ImportError:
    # Fallback to basic dataset
    AugmentedMultiTaskDataset = MultiTaskCVSDataset


def main(config_path: str = "configs/exp12_regularized.yaml"):
    """Main training function."""
    config = load_config(config_path)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["data"]["results_dir"]) / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(results_dir), "train")
    logger.info(f"Config: {config}")
    logger.info(f"Device: {device}")
    logger.info(f"Results: {results_dir}")

    # Get augmentation config
    aug_config = config.get("augmentation", {})
    training_cfg = config["training"]
    lora_cfg = config["lora"]

    # Log new regularization settings
    logger.info("\n" + "=" * 70)
    logger.info("Regularization Settings (Exp12)")
    logger.info("=" * 70)
    logger.info(f"Weight decay: {training_cfg['weight_decay']}")
    logger.info(f"Label smoothing: {training_cfg.get('label_smoothing', 0.0)}")
    logger.info(f"MixUp alpha: {aug_config.get('mixup_alpha', 0.0)}, prob: {aug_config.get('mixup_prob', 0.0)}")
    logger.info(f"CutMix alpha: {aug_config.get('cutmix_alpha', 0.0)}, prob: {aug_config.get('cutmix_prob', 0.0)}")

    ham_config = config.get("hard_attention_masking", {})
    logger.info(f"Hard attention masking: enabled={ham_config.get('enabled', False)}, "
                f"train={ham_config.get('apply_during_training', False)}, "
                f"eval={ham_config.get('apply_during_eval', False)}")

    # Create datasets
    logger.info("\nCreating datasets...")

    # Check if SAGES dataset is configured
    use_combined = config["data"].get("sages_root") is not None
    active_collate_fn = combined_collate_fn if use_combined else collate_fn

    if use_combined:
        logger.info("Using combined Endoscapes + SAGES dataset")
        train_dataset = CombinedMultiTaskDataset(
            endoscapes_root=config["data"]["endoscapes_root"],
            sages_root=config["data"]["sages_root"],
            split="train",
            num_frames=config["dataset"]["num_frames"],
            resolution=config["dataset"]["resolution"],
            mask_resolution=config["dataset"].get("mask_resolution", 64),
            augment=True,
            horizontal_flip_prob=aug_config.get("horizontal_flip_prob", 0.5),
            use_synthetic_masks=config["dataset"].get("use_synthetic_masks", True),
            endoscapes_gt_masks_dir=config["data"].get("gt_masks_dir"),
            endoscapes_synthetic_masks_dir=config["data"].get("endoscapes_synthetic_masks_dir"),
            sages_masks_dir=config["data"].get("sages_masks_dir"),
            include_endoscapes=config["data"].get("include_endoscapes", True),
            include_sages=config["data"].get("include_sages", True),
        )

        val_dataset = CombinedMultiTaskDataset(
            endoscapes_root=config["data"]["endoscapes_root"],
            sages_root=config["data"]["sages_root"],
            split="val",
            num_frames=config["dataset"]["num_frames"],
            resolution=config["dataset"]["resolution"],
            mask_resolution=config["dataset"].get("mask_resolution", 64),
            augment=False,
            use_synthetic_masks=config["dataset"].get("use_synthetic_masks", True),
            endoscapes_gt_masks_dir=config["data"].get("gt_masks_dir"),
            endoscapes_synthetic_masks_dir=config["data"].get("endoscapes_synthetic_masks_dir"),
            sages_masks_dir=config["data"].get("sages_masks_dir"),
            include_endoscapes=config["data"].get("include_endoscapes", True),
            include_sages=config["data"].get("include_sages", True),
        )
    else:
        logger.info("Using Endoscapes-only dataset")
        train_dataset = AugmentedMultiTaskDataset(
            root_dir=config["data"]["endoscapes_root"],
            split="train",
            num_frames=config["dataset"]["num_frames"],
            resolution=config["dataset"]["resolution"],
            mask_resolution=config["dataset"].get("mask_resolution", 64),
            augment=True,
            use_synthetic_masks=config["dataset"].get("use_synthetic_masks", True),
            gt_masks_dir=config["data"].get("gt_masks_dir"),
            synthetic_masks_dir=config["data"].get("synthetic_masks_dir"),
            horizontal_flip_prob=aug_config.get("horizontal_flip_prob", 0.5),
            rotation_degrees=aug_config.get("rotation_degrees", 15.0),
            color_jitter=aug_config.get("color_jitter"),
            random_erasing_prob=aug_config.get("random_erasing_prob", 0.2),
            gaussian_blur_prob=aug_config.get("gaussian_blur_prob", 0.1),
            gaussian_blur_sigma=tuple(aug_config.get("gaussian_blur_sigma", [0.1, 2.0])),
        )

        val_dataset = MultiTaskCVSDataset(
            root_dir=config["data"]["endoscapes_root"],
            split="val",
            num_frames=config["dataset"]["num_frames"],
            resolution=config["dataset"]["resolution"],
            mask_resolution=config["dataset"].get("mask_resolution", 64),
            augment=False,
            use_synthetic_masks=config["dataset"].get("use_synthetic_masks", True),
            gt_masks_dir=config["data"].get("gt_masks_dir"),
            synthetic_masks_dir=config["data"].get("synthetic_masks_dir"),
        )

    logger.info(f"Train: {len(train_dataset)} clips")
    logger.info(f"Val: {len(val_dataset)} clips")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["num_workers"],
        collate_fn=active_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        collate_fn=active_collate_fn,
        pin_memory=True,
    )

    # Create model with LoRA
    logger.info("\n" + "=" * 70)
    logger.info("Creating V-JEPA model with LoRA adapters")
    logger.info("=" * 70)

    model_cfg = config["model"]
    model = VJEPA_LoRA(
        model_name=model_cfg["name"],
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        lora_r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg.get("target_modules"),
        cvs_hidden=model_cfg.get("cvs_hidden", 512),
        cvs_dropout=model_cfg.get("cvs_dropout", 0.5),
        attention_heads=model_cfg.get("attention_heads", 8),
        attention_dropout=model_cfg.get("attention_dropout", 0.1),
        num_seg_classes=model_cfg.get("num_seg_classes", 5),
        seg_output_size=model_cfg.get("seg_output_size", 64),
        seg_dropout=model_cfg.get("seg_dropout", 0.1),
    )
    model = model.to(device)

    total_params = model.get_num_total_params()
    trainable_params = model.get_num_trainable_params()
    logger.info(f"Total params: {total_params / 1e6:.1f}M")
    logger.info(f"Trainable params: {trainable_params / 1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")

    # Create optimizer with separate LRs for LoRA and heads
    lora_params = model.get_lora_params()
    head_params = model.get_head_params()

    logger.info(f"LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.2f}M")
    logger.info(f"Head params: {sum(p.numel() for p in head_params) / 1e6:.2f}M")

    param_groups = [
        {'params': lora_params, 'lr': training_cfg["lora_lr"]},
        {'params': head_params, 'lr': training_cfg["head_lr"]},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=training_cfg["weight_decay"],
    )
    logger.info(f"Optimizer: LoRA_lr={training_cfg['lora_lr']}, head_lr={training_cfg['head_lr']}")

    # Create scheduler
    accum_steps = training_cfg.get("gradient_accumulation", 1)
    steps_per_epoch = len(train_loader) // accum_steps
    num_training_steps = steps_per_epoch * training_cfg["epochs"]
    warmup_ratio = training_cfg.get("warmup_epochs", 0.5) / training_cfg["epochs"]

    scheduler = create_cosine_scheduler_with_warmup(
        optimizer,
        num_training_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=0.01,
    )
    logger.info(f"Scheduler: cosine with {warmup_ratio*100:.1f}% warmup")

    # Loss
    criterion = MultiTaskLoss(
        cvs_weight=config["loss"]["cvs_weight"],
        seg_weight=config["loss"]["seg_weight"],
        cvs_pos_weight=config["loss"].get("cvs_pos_weight"),
        seg_class_weights=config["loss"].get("seg_class_weights"),
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=training_cfg["early_stopping_patience"],
        mode="max",
    )

    # Mixed precision
    scaler = GradScaler() if training_cfg.get("mixed_precision", True) else None

    # Training config for epoch — includes augmentation and hard attention masking
    epoch_config = {
        "gradient_accumulation": accum_steps,
        "mixed_precision": training_cfg.get("mixed_precision", True),
        "grad_clip": training_cfg.get("grad_clip", 1.0),
        "augmentation": aug_config,
        "hard_attention_masking": ham_config,
        "label_smoothing": training_cfg.get("label_smoothing", 0.0),
    }

    best_metric = 0.0
    best_epoch = 0

    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("Starting Regularized LoRA Fine-Tuning (Exp12)")
    logger.info("=" * 70)

    for epoch in range(1, training_cfg["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{training_cfg['epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch_config, scaler, logger
        )
        logger.info(f"Train | mAP: {train_metrics['mAP']*100:.2f}% | Loss: {train_metrics['loss']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch_config)
        logger.info(
            f"Val   | mAP: {val_metrics['mAP']*100:.2f}% | "
            f"AP: C1={val_metrics['AP_C1']*100:.2f}%, C2={val_metrics['AP_C2']*100:.2f}%, C3={val_metrics['AP_C3']*100:.2f}%"
        )

        # Check best
        current_metric = val_metrics["mAP"]
        is_best = current_metric > best_metric

        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            logger.info(f"New best mAP: {best_metric*100:.2f}%")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": config,
        }

        save_checkpoint(
            checkpoint,
            str(results_dir / f"epoch_{epoch}.pt"),
            is_best=is_best,
            best_path=str(results_dir / "best_model.pt"),
        )

        # Early stopping
        if early_stopping(current_metric):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best mAP: {best_metric*100:.2f}% at epoch {best_epoch}")
    logger.info(f"Final model saved to: {results_dir / 'best_model.pt'}")

    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regularized LoRA fine-tuning for V-JEPA CVS model (Exp12)")
    parser.add_argument("--config", type=str, default="configs/exp12_regularized.yaml")
    args = parser.parse_args()

    main(args.config)
