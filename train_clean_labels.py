"""
Clean Labels Training Script (Exp14)
=====================================
V-JEPA LoRA fine-tuning with:
  - Clean labels via ImprovedMultiTaskCVSDataset (5 frames, majority vote)
  - Centre crop 480x480 to remove endoscopic black borders
  - LoRA on later layers only (18-23 of 24)
  - All Exp12 regularization (MixUp, CutMix, label smoothing, HAM)

Based on train_regularized.py (Exp12) with targeted fixes for data quality.
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

from dataset_improved import ImprovedMultiTaskCVSDataset, collate_fn
from utils import (
    AverageMeter,
    EarlyStopping,
    compute_metrics,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logging,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: peft not installed. Run: pip install peft")


# ============================================================================
# Augmentation / regularization (same as train_regularized.py)
# ============================================================================

def mixup_data(videos, labels, alpha=0.8):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = videos.size(0)
    index = torch.randperm(batch_size, device=videos.device)
    mixed_videos = lam * videos + (1 - lam) * videos[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_videos, mixed_labels, lam


def cutmix_data(videos, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = videos.size(0)
    index = torch.randperm(batch_size, device=videos.device)
    _, T, C, H, W = videos.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    mixed_videos = videos.clone()
    mixed_videos[:, :, :, y1:y2, x1:x2] = videos[index, :, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (H * W)
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_videos, mixed_labels, lam


def apply_batch_augmentation(videos, labels, aug_config):
    mixup_prob = aug_config.get("mixup_prob", 0.0)
    cutmix_prob = aug_config.get("cutmix_prob", 0.0)
    r = np.random.rand()
    if r < mixup_prob:
        videos, labels, lam = mixup_data(videos, labels, aug_config.get("mixup_alpha", 0.8))
        return videos, labels, lam, True
    elif r < mixup_prob + cutmix_prob:
        videos, labels, lam = cutmix_data(videos, labels, aug_config.get("cutmix_alpha", 1.0))
        return videos, labels, lam, True
    return videos, labels, 1.0, False


def smooth_labels(labels, smoothing=0.1):
    return labels * (1.0 - smoothing) + (1.0 - labels) * smoothing


def create_spatial_mask_from_segmentation(
    seg_masks, mask_batch_indices, batch_size,
    spatial_size=16, num_temporal_bins=8,
):
    spatial_tokens = spatial_size * spatial_size
    num_tokens = num_temporal_bins * spatial_tokens
    device = seg_masks.device
    token_mask = torch.ones(batch_size, num_tokens, device=device)
    if len(mask_batch_indices) == 0:
        return token_mask
    for b_idx in torch.unique(mask_batch_indices):
        b = b_idx.item()
        batch_masks = seg_masks[mask_batch_indices == b_idx]
        mask = batch_masks[0].float().unsqueeze(0).unsqueeze(0)
        downsampled = F.interpolate(
            mask, size=(spatial_size, spatial_size), mode="nearest"
        ).squeeze()
        spatial_mask = (downsampled > 0).float().view(-1)
        tiled_mask = spatial_mask.unsqueeze(0).expand(num_temporal_bins, -1).reshape(-1)
        token_mask[b] = tiled_mask
    return token_mask


def apply_hard_attention_mask(features, token_mask):
    return features * token_mask.unsqueeze(-1)


# ============================================================================
# Model classes
# ============================================================================

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x):
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        attn_out, _ = self.attention(query, x, x)
        return self.norm(attn_out.squeeze(1))


class LightweightSegDecoder(nn.Module):
    def __init__(self, hidden_dim=1024, num_classes=5, output_size=64, dropout=0.1):
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

    def forward(self, features, frame_indices, batch_indices,
                num_temporal_bins=8, spatial_size=16):
        if len(frame_indices) == 0:
            return torch.zeros(
                0, self.num_classes, self.output_size, self.output_size,
                device=features.device,
            )
        B, num_tokens, D = features.shape
        spatial_tokens = spatial_size * spatial_size
        temporal_indices = frame_indices // 2
        frame_features = []
        for batch_idx, temp_idx in zip(batch_indices, temporal_indices):
            start_idx = temp_idx * spatial_tokens
            end_idx = start_idx + spatial_tokens
            if end_idx > num_tokens:
                end_idx = num_tokens
                start_idx = max(0, end_idx - spatial_tokens)
            spatial_feats = features[batch_idx, start_idx:end_idx]
            if spatial_feats.shape[0] < spatial_tokens:
                padding = torch.zeros(
                    spatial_tokens - spatial_feats.shape[0], D,
                    device=features.device,
                )
                spatial_feats = torch.cat([spatial_feats, padding], dim=0)
            frame_features.append(spatial_feats)
        frame_features = torch.stack(frame_features, dim=0)
        frame_features = self.proj(frame_features)
        N = frame_features.shape[0]
        frame_features = frame_features.view(N, spatial_size, spatial_size, -1)
        frame_features = frame_features.permute(0, 3, 1, 2).contiguous()
        return self.decoder(frame_features)


class VJEPA_LoRA_LayerSelective(nn.Module):
    """
    V-JEPA with LoRA adapters on *selected* layers only.

    Uses PEFT's native layers_to_transform parameter for clean layer selection.

    V-JEPA ViT-L module naming:
        encoder.layer.{0-23}.attention.query   (nn.Linear, 1024x1024)
        encoder.layer.{0-23}.attention.key     (nn.Linear, 1024x1024)
        encoder.layer.{0-23}.attention.value   (nn.Linear, 1024x1024)

    NOTE: V-JEPA uses "query"/"key"/"value", NOT "q_proj"/"k_proj"/"v_proj".
    """

    # V-JEPA attention projection names (not q_proj/k_proj/v_proj!)
    VJEPA_ATTN_MODULES = ["query", "key", "value"]

    def __init__(
        self,
        model_name="facebook/vjepa2-vitl-fpc16-256-ssv2",
        hidden_dim=1024,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=None,
        target_layers=None,
        cvs_hidden=512,
        cvs_dropout=0.5,
        attention_heads=8,
        attention_dropout=0.1,
        num_seg_classes=5,
        seg_output_size=64,
        seg_dropout=0.1,
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
        backbone = backbone.float()

        # Freeze everything
        for param in backbone.parameters():
            param.requires_grad = False

        # Map config names to V-JEPA's actual module names
        # Config may use q_proj/k_proj/v_proj (common convention) but
        # V-JEPA uses query/key/value
        ALIAS_MAP = {
            "q_proj": "query", "k_proj": "key", "v_proj": "value",
            "query": "query", "key": "key", "value": "value",
        }
        if target_modules is None:
            target_modules = ["query", "value"]
        resolved_modules = [ALIAS_MAP.get(m, m) for m in target_modules]

        # Verify modules exist in the model
        found = set()
        for name, module in backbone.named_modules():
            if isinstance(module, nn.Linear):
                for tm in resolved_modules:
                    if name.endswith(f".{tm}"):
                        found.add(tm)
        print(f"Resolved target modules: {resolved_modules}")
        print(f"  Verified in model: {sorted(found)}")

        # Build LoRA config with PEFT's native layer selection
        lora_kwargs = dict(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=resolved_modules,
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=[],
        )

        if target_layers is not None:
            # PEFT's layers_to_transform selects layers by index.
            # layers_pattern tells PEFT how to parse layer numbers from names.
            # V-JEPA: "encoder.layer.{N}.attention.query" -> pattern is "layer"
            lora_kwargs["layers_to_transform"] = target_layers
            lora_kwargs["layers_pattern"] = "layer"
            print(f"Layer-selective LoRA: layers {target_layers} (6 of 24)")
        else:
            print("LoRA applied to ALL 24 layers")

        lora_config = LoraConfig(**lora_kwargs)
        print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}")
        self.backbone = get_peft_model(backbone, lora_config)
        self.backbone.print_trainable_parameters()

        # CVS head
        self.pooler = AttentionPooling(hidden_dim, attention_heads, attention_dropout)
        self.cvs_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, cvs_hidden),
            nn.GELU(),
            nn.Dropout(cvs_dropout),
            nn.Linear(cvs_hidden, 3),
        )

        # Seg head
        self.seg_head = LightweightSegDecoder(
            hidden_dim, num_seg_classes, seg_output_size, seg_dropout,
        )

        self._init_heads()

    def _init_heads(self):
        for module in self.cvs_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pixel_values_videos, mask_frame_indices=None,
                mask_batch_indices=None, attention_mask=None):
        features = self.backbone.get_vision_features(
            pixel_values_videos=pixel_values_videos
        )
        features = features.float()

        if attention_mask is not None:
            features = apply_hard_attention_mask(features, attention_mask)

        pooled = self.pooler(features)
        cvs_logits = self.cvs_head(pooled)
        result = {"cvs_logits": cvs_logits}

        if (mask_frame_indices is not None and mask_batch_indices is not None
                and len(mask_frame_indices) > 0):
            seg_logits = self.seg_head(features, mask_frame_indices, mask_batch_indices)
            result["seg_logits"] = seg_logits

        return result

    # V-JEPA fpc16 expects exactly 16 frames
    EXPECTED_FRAMES = 16

    def _pad_to_expected_frames(self, video_tensor):
        """
        Pad video to EXPECTED_FRAMES by repeating the last frame.

        V-JEPA fpc16 expects exactly 16 frames. When using fewer frames
        (e.g. 5), we repeat the last frame to fill the temporal dimension.
        This preserves the spatial content of the actual frames while
        giving the model a full-length input.

        Args:
            video_tensor: [B, T, C, H, W] normalized tensor
        Returns:
            [B, 16, C, H, W] padded tensor
        """
        B, T, C, H, W = video_tensor.shape
        if T >= self.EXPECTED_FRAMES:
            return video_tensor[:, :self.EXPECTED_FRAMES]
        # Repeat last frame to fill
        pad_count = self.EXPECTED_FRAMES - T
        last_frame = video_tensor[:, -1:].expand(B, pad_count, C, H, W)
        return torch.cat([video_tensor, last_frame], dim=1)

    def process_videos(self, videos, device):
        """
        Process raw videos to V-JEPA input format.

        Handles variable frame counts by padding to 16 frames (V-JEPA fpc16).
        """
        if isinstance(videos, torch.Tensor):
            video_tensor = videos
            if video_tensor.dim() == 5:
                B, T, dim3, H, W = video_tensor.shape
                if dim3 == 3 and video_tensor.dtype == torch.float32:
                    if video_tensor.min() < -0.5 or video_tensor.max() > 1.5:
                        return self._pad_to_expected_frames(video_tensor.to(device))
                    video_tensor = video_tensor.to(device)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
                    normalized = (video_tensor - mean) / std
                    return self._pad_to_expected_frames(normalized)
                elif W == 3:
                    video_tensor = video_tensor.permute(0, 1, 4, 2, 3).contiguous()
                    video_tensor = video_tensor.to(device).float() / 255.0
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
                    normalized = (video_tensor - mean) / std
                    return self._pad_to_expected_frames(normalized)
            raise ValueError(f"Unexpected tensor shape: {video_tensor.shape}")

        elif isinstance(videos, list):
            if not videos:
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
            normalized = (video_tensor - mean) / std
            return self._pad_to_expected_frames(normalized)

        raise ValueError(f"Unsupported videos type: {type(videos)}")

    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_lora_params(self):
        return [p for n, p in self.backbone.named_parameters()
                if p.requires_grad and 'lora' in n.lower()]

    def get_head_params(self):
        head_params = []
        for m in [self.pooler, self.cvs_head, self.seg_head]:
            head_params.extend(p for p in m.parameters() if p.requires_grad)
        return head_params


# ============================================================================
# Loss and metrics (same as train_regularized.py)
# ============================================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, cvs_weight=1.0, seg_weight=0.3,
                 cvs_pos_weight=None, seg_class_weights=None):
        super().__init__()
        self.cvs_weight = cvs_weight
        self.seg_weight = seg_weight
        if cvs_pos_weight is not None:
            self.cvs_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(cvs_pos_weight, dtype=torch.float32)
            )
        else:
            self.cvs_loss = nn.BCEWithLogitsLoss()
        if seg_class_weights is not None:
            self.seg_loss = nn.CrossEntropyLoss(
                weight=torch.tensor(seg_class_weights, dtype=torch.float32),
                ignore_index=255,
            )
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
            total = self.cvs_weight * cvs_loss + self.seg_weight * seg_loss
        else:
            seg_loss = torch.tensor(0.0, device=cvs_logits.device)
            total = self.cvs_weight * cvs_loss
        return {"total_loss": total, "cvs_loss": cvs_loss, "seg_loss": seg_loss}


def compute_seg_metrics(seg_logits, seg_labels, num_classes=5):
    if seg_logits.shape[0] == 0:
        return {"seg_miou": 0.0, "seg_acc": 0.0}
    preds = seg_logits.argmax(dim=1).view(-1)
    labels = seg_labels.view(-1)
    valid = labels != 255
    preds, labels = preds[valid], labels[valid]
    if len(labels) == 0:
        return {"seg_miou": 0.0, "seg_acc": 0.0}
    acc = (preds == labels).float().mean().item()
    ious = []
    for c in range(num_classes):
        inter = ((preds == c) & (labels == c)).sum().float()
        union = ((preds == c) | (labels == c)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return {"seg_miou": np.mean(ious) if ious else 0.0, "seg_acc": acc}


def create_cosine_scheduler_with_warmup(optimizer, num_training_steps,
                                         warmup_ratio=0.1, min_lr_ratio=0.01):
    num_warmup = int(num_training_steps * warmup_ratio)

    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        progress = float(step - num_warmup) / float(max(1, num_training_steps - num_warmup))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Augmented dataset wrapper (adds per-frame augmentation to improved dataset)
# ============================================================================

class AugmentedImprovedDataset(ImprovedMultiTaskCVSDataset):
    """
    Wraps ImprovedMultiTaskCVSDataset with per-frame augmentation
    (color jitter, random erasing, gaussian blur, rotation).

    Centre crop is handled by the base class. This adds on-the-fly transforms.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        # Augmentation params
        rotation_degrees=15.0,
        color_jitter=None,
        random_erasing_prob=0.2,
        gaussian_blur_prob=0.1,
        gaussian_blur_sigma=(0.1, 2.0),
        # Pass all other kwargs to base
        **kwargs,
    ):
        super().__init__(root_dir=root_dir, split=split, **kwargs)
        self.rotation_degrees = rotation_degrees
        self.color_jitter_cfg = color_jitter or {}
        self.random_erasing_prob = random_erasing_prob
        self.gaussian_blur_prob = gaussian_blur_prob
        self.gaussian_blur_sigma = gaussian_blur_sigma

    def _apply_color_jitter(self, frame):
        if not self.color_jitter_cfg:
            return frame
        frame = frame.astype(np.float32)
        if 'brightness' in self.color_jitter_cfg:
            f = 1.0 + np.random.uniform(
                -self.color_jitter_cfg['brightness'],
                self.color_jitter_cfg['brightness'],
            )
            frame = frame * f
        if 'contrast' in self.color_jitter_cfg:
            f = 1.0 + np.random.uniform(
                -self.color_jitter_cfg['contrast'],
                self.color_jitter_cfg['contrast'],
            )
            frame = (frame - frame.mean()) * f + frame.mean()
        if 'saturation' in self.color_jitter_cfg:
            f = 1.0 + np.random.uniform(
                -self.color_jitter_cfg['saturation'],
                self.color_jitter_cfg['saturation'],
            )
            gray = frame.mean(axis=-1, keepdims=True)
            frame = frame * f + gray * (1 - f)
        if 'hue' in self.color_jitter_cfg:
            shift = np.random.uniform(
                -self.color_jitter_cfg['hue'],
                self.color_jitter_cfg['hue'],
            )
            frame = np.roll(frame, int(shift * frame.shape[-1]), axis=-1)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _apply_random_erasing(self, frame):
        if np.random.random() > self.random_erasing_prob:
            return frame
        h, w = frame.shape[:2]
        area_ratio = np.random.uniform(0.05, 0.20)
        aspect = np.random.uniform(0.5, 2.0)
        area = h * w * area_ratio
        ph, pw = int(np.sqrt(area / aspect)), int(np.sqrt(area * aspect))
        ph, pw = min(ph, h - 1), min(pw, w - 1)
        top = np.random.randint(0, h - ph)
        left = np.random.randint(0, w - pw)
        frame = frame.copy()
        frame[top:top + ph, left:left + pw] = np.random.randint(0, 255, (ph, pw, 3))
        return frame

    def _apply_gaussian_blur(self, frame):
        if np.random.random() > self.gaussian_blur_prob:
            return frame
        from scipy.ndimage import gaussian_filter
        sigma = np.random.uniform(*self.gaussian_blur_sigma)
        frame = gaussian_filter(frame.astype(np.float32), sigma=(sigma, sigma, 0))
        return np.clip(frame, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if not self.augment:
            return sample

        video = sample["video"]  # (T, H, W, C)
        augmented = []
        for t in range(len(video)):
            frame = video[t]
            frame = self._apply_color_jitter(frame)
            frame = self._apply_random_erasing(frame)
            frame = self._apply_gaussian_blur(frame)
            augmented.append(frame)

        sample["video"] = np.stack(augmented, axis=0)
        return sample


# ============================================================================
# Train / validate (same logic as train_regularized.py)
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scheduler,
                device, config, scaler, logger):
    model.train()
    loss_meter = AverageMeter()
    cvs_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    all_preds, all_targets = [], []

    accum_steps = config.get("gradient_accumulation", 1)
    use_amp = config.get("mixed_precision", True)
    aug_config = config.get("augmentation", {})
    ham_config = config.get("hard_attention_masking", {})
    label_smoothing = config.get("label_smoothing", 0.0)
    ham_enabled = ham_config.get("enabled", False) and ham_config.get("apply_during_training", False)
    ham_spatial_size = ham_config.get("spatial_size", 16)

    # Infer temporal bins from config
    num_frames = config.get("num_frames", 16)
    # V-JEPA fpc=16: temporal stride of 2 -> bins = ceil(T / 2)
    # But for T < 16 the model may produce fewer tokens
    num_temporal_bins = config.get("num_temporal_bins", None)

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        videos = batch["videos"]
        cvs_labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)
        mask_frame_indices = batch["mask_frame_indices"].to(device)
        mask_batch_indices = batch["mask_batch_indices"].to(device)

        pixel_values = model.process_videos(videos, device)
        original_cvs_labels = cvs_labels.clone()

        pixel_values, cvs_labels, lam, augmented = apply_batch_augmentation(
            pixel_values, cvs_labels, aug_config
        )

        if label_smoothing > 0:
            cvs_labels = smooth_labels(cvs_labels, label_smoothing)

        # Hard attention mask
        attention_mask = None
        if ham_enabled and len(mask_batch_indices) > 0:
            # Determine temporal bins from actual feature size on first batch
            if num_temporal_bins is None:
                # Will be set after first forward pass; skip HAM for first batch
                pass
            else:
                attention_mask = create_spatial_mask_from_segmentation(
                    masks, mask_batch_indices,
                    batch_size=pixel_values.size(0),
                    spatial_size=ham_spatial_size,
                    num_temporal_bins=num_temporal_bins,
                )

        is_accumulating = (
            (batch_idx + 1) % accum_steps != 0
            and (batch_idx + 1 < len(train_loader))
        )

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(pixel_values, mask_frame_indices, mask_batch_indices, attention_mask)

                # Infer temporal bins from feature shape on first pass
                if num_temporal_bins is None and "cvs_logits" in outputs:
                    features_tokens = outputs["cvs_logits"].shape  # Not useful
                    # Get from features directly in next iteration
                    # For now use ceil(T/2) heuristic
                    config["num_temporal_bins"] = max(1, (num_frames + 1) // 2)
                    num_temporal_bins = config["num_temporal_bins"]

                seg_logits = outputs.get("seg_logits")
                loss_dict = criterion(
                    outputs["cvs_logits"], cvs_labels,
                    seg_logits, masks if seg_logits is not None else None,
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
            if num_temporal_bins is None:
                config["num_temporal_bins"] = max(1, (num_frames + 1) // 2)
                num_temporal_bins = config["num_temporal_bins"]
            seg_logits = outputs.get("seg_logits")
            loss_dict = criterion(
                outputs["cvs_logits"], cvs_labels,
                seg_logits, masks if seg_logits is not None else None,
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
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_targets = [], []
    seg_miou_sum, seg_count = 0.0, 0

    for batch in tqdm(val_loader, desc="Validating"):
        videos = batch["videos"]
        cvs_labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)
        mask_frame_indices = batch["mask_frame_indices"].to(device)
        mask_batch_indices = batch["mask_batch_indices"].to(device)

        pixel_values = model.process_videos(videos, device)
        outputs = model(pixel_values, mask_frame_indices, mask_batch_indices)

        seg_logits = outputs.get("seg_logits")
        loss_dict = criterion(
            outputs["cvs_logits"], cvs_labels,
            seg_logits, masks if seg_logits is not None else None,
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


# ============================================================================
# Main
# ============================================================================

def main(config_path="configs/exp14_clean_labels.yaml"):
    config = load_config(config_path)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["data"]["results_dir"]) / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(results_dir), "train")
    logger.info(f"Config: {config}")
    logger.info(f"Device: {device}")
    logger.info(f"Results: {results_dir}")

    aug_config = config.get("augmentation", {})
    training_cfg = config["training"]
    lora_cfg = config["lora"]
    dataset_cfg = config["dataset"]

    # Log experiment settings
    logger.info("\n" + "=" * 70)
    logger.info("Exp14: Clean Labels + All Supervisor Fixes")
    logger.info("=" * 70)
    logger.info(f"  num_frames:      {dataset_cfg['num_frames']}")
    logger.info(f"  frame_step:      {dataset_cfg.get('frame_step', 1)}")
    logger.info(f"  centre_crop:     {dataset_cfg.get('centre_crop', None)}")
    logger.info(f"  label_strategy:  {dataset_cfg.get('label_strategy', 'center')}")
    logger.info(f"  clip_subsample:  {dataset_cfg.get('clip_subsample', 1)}")
    logger.info(f"  LoRA layers:     {lora_cfg.get('target_layers', 'all')}")
    logger.info(f"  Weight decay:    {training_cfg['weight_decay']}")
    logger.info(f"  Label smoothing: {training_cfg.get('label_smoothing', 0.0)}")
    logger.info(f"  MixUp prob:      {aug_config.get('mixup_prob', 0.0)}")
    logger.info(f"  CutMix prob:     {aug_config.get('cutmix_prob', 0.0)}")

    ham_config = config.get("hard_attention_masking", {})
    logger.info(f"  HAM: enabled={ham_config.get('enabled', False)}, "
                f"train={ham_config.get('apply_during_training', False)}")

    # Create datasets using ImprovedMultiTaskCVSDataset
    logger.info("\nCreating improved datasets...")

    shared_ds_kwargs = dict(
        num_frames=dataset_cfg["num_frames"],
        frame_step=dataset_cfg.get("frame_step", 1),
        resolution=dataset_cfg["resolution"],
        centre_crop=dataset_cfg.get("centre_crop", 480),
        mask_resolution=dataset_cfg.get("mask_resolution", 64),
        use_synthetic_masks=dataset_cfg.get("use_synthetic_masks", True),
        gt_masks_dir=config["data"].get("gt_masks_dir"),
        synthetic_masks_dir=config["data"].get("synthetic_masks_dir"),
        label_strategy=dataset_cfg.get("label_strategy", "majority"),
        clip_subsample=dataset_cfg.get("clip_subsample", 1),
        binarize_threshold=dataset_cfg.get("binarize_threshold", 0.5),
    )

    train_dataset = AugmentedImprovedDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="train",
        augment=True,
        horizontal_flip_prob=aug_config.get("horizontal_flip_prob", 0.5),
        rotation_degrees=aug_config.get("rotation_degrees", 15.0),
        color_jitter=aug_config.get("color_jitter"),
        random_erasing_prob=aug_config.get("random_erasing_prob", 0.2),
        gaussian_blur_prob=aug_config.get("gaussian_blur_prob", 0.1),
        gaussian_blur_sigma=tuple(aug_config.get("gaussian_blur_sigma", [0.1, 2.0])),
        **shared_ds_kwargs,
    )

    val_dataset = ImprovedMultiTaskCVSDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="val",
        augment=False,
        **shared_ds_kwargs,
    )

    logger.info(f"Train: {len(train_dataset)} clips")
    logger.info(f"Val:   {len(val_dataset)} clips")

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model with layer-selective LoRA
    logger.info("\n" + "=" * 70)
    logger.info("Creating V-JEPA model with layer-selective LoRA")
    logger.info("=" * 70)

    model_cfg = config["model"]
    model = VJEPA_LoRA_LayerSelective(
        model_name=model_cfg["name"],
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        lora_r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg.get("target_modules"),
        target_layers=lora_cfg.get("target_layers"),
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
    logger.info(f"Trainable params: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")

    # Optimizer with separate LRs
    lora_params = model.get_lora_params()
    head_params = model.get_head_params()
    logger.info(f"LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.2f}M")
    logger.info(f"Head params: {sum(p.numel() for p in head_params) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": training_cfg["lora_lr"]},
            {"params": head_params, "lr": training_cfg["head_lr"]},
        ],
        weight_decay=training_cfg["weight_decay"],
    )
    logger.info(f"Optimizer: LoRA_lr={training_cfg['lora_lr']}, head_lr={training_cfg['head_lr']}")

    # Scheduler
    accum_steps = training_cfg.get("gradient_accumulation", 1)
    steps_per_epoch = len(train_loader) // accum_steps
    num_training_steps = steps_per_epoch * training_cfg["epochs"]
    warmup_ratio = training_cfg.get("warmup_epochs", 0.5) / training_cfg["epochs"]

    scheduler = create_cosine_scheduler_with_warmup(
        optimizer, num_training_steps,
        warmup_ratio=warmup_ratio, min_lr_ratio=0.01,
    )

    # Loss
    criterion = MultiTaskLoss(
        cvs_weight=config["loss"]["cvs_weight"],
        seg_weight=config["loss"]["seg_weight"],
        cvs_pos_weight=config["loss"].get("cvs_pos_weight"),
        seg_class_weights=config["loss"].get("seg_class_weights"),
    )

    early_stopping = EarlyStopping(
        patience=training_cfg["early_stopping_patience"], mode="max",
    )
    scaler = GradScaler() if training_cfg.get("mixed_precision", True) else None

    # Epoch config
    epoch_config = {
        "gradient_accumulation": accum_steps,
        "mixed_precision": training_cfg.get("mixed_precision", True),
        "grad_clip": training_cfg.get("grad_clip", 1.0),
        "augmentation": aug_config,
        "hard_attention_masking": ham_config,
        "label_smoothing": training_cfg.get("label_smoothing", 0.0),
        "num_frames": dataset_cfg["num_frames"],
        "num_temporal_bins": None,  # Auto-detect from model
    }

    best_metric, best_epoch = 0.0, 0

    logger.info("\n" + "=" * 70)
    logger.info("Starting Exp14: Clean Labels Training")
    logger.info("=" * 70)

    for epoch in range(1, training_cfg["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{training_cfg['epochs']}")

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch_config, scaler, logger,
        )
        logger.info(
            f"Train | mAP: {train_metrics['mAP'] * 100:.2f}% | "
            f"Loss: {train_metrics['loss']:.4f}"
        )

        val_metrics = validate(model, val_loader, criterion, device, epoch_config)
        logger.info(
            f"Val   | mAP: {val_metrics['mAP'] * 100:.2f}% | "
            f"AP: C1={val_metrics['AP_C1'] * 100:.2f}%, "
            f"C2={val_metrics['AP_C2'] * 100:.2f}%, "
            f"C3={val_metrics['AP_C3'] * 100:.2f}%"
        )

        current_metric = val_metrics["mAP"]
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            logger.info(f"*** New best mAP: {best_metric * 100:.2f}%")

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "config": config,
            },
            str(results_dir / f"epoch_{epoch}.pt"),
            is_best=is_best,
            best_path=str(results_dir / "best_model.pt"),
        )

        if early_stopping(current_metric):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best mAP: {best_metric * 100:.2f}% at epoch {best_epoch}")
    logger.info(f"Model saved to: {results_dir / 'best_model.pt'}")

    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp14: Clean labels + LoRA on later layers"
    )
    parser.add_argument(
        "--config", type=str, default="configs/exp14_clean_labels.yaml"
    )
    args = parser.parse_args()
    main(args.config)
