"""
Experiment 11: Attention-Supervised LoRA Training
==================================================
Train V-JEPA with LoRA + attention supervision using segmentation masks.

Key innovation: Add loss to make V-JEPA attend to anatomical regions
instead of uniformly across all patches.

From Exp10b analysis:
- V-JEPA attention entropy: ~98% (nearly uniform)
- LoRA (even with k_proj) did NOT change this
- Performance gains came from value extraction, not attention focus

This experiment adds explicit attention supervision:
- Create target attention from segmentation masks
- Penalize uniform attention, reward anatomy focus
- Goal: Reduce entropy from 98% to <90%
"""

import argparse
import datetime
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_multitask import MultiTaskCVSDataset
from utils import load_config


def custom_collate_fn(batch):
    """
    Custom collate function that handles samples with/without masks.

    Problem: Some samples have empty masks [0, 64, 64] while others have valid
    masks [N, 64, 64]. PyTorch's default collate can't stack these together.

    Solution: Pad empty masks with zeros and track which samples have valid masks.

    Dataset returns:
        - video: numpy array (T, H, W, C) uint8
        - labels: tensor (3,) float32
        - masks: tensor (N, H, W) int64 - may be empty [0, H, W]
        - mask_indices: tensor (N,) int64
        - has_masks: bool
        - meta: dict
    """
    # Convert numpy video to tensor and stack
    videos = []
    for b in batch:
        video = b['video']
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        videos.append(video)
    videos = torch.stack(videos)

    # Labels are already tensors
    labels = torch.stack([b['labels'] for b in batch])

    # Handle masks - find max number of masks in batch
    max_masks = 0
    for b in batch:
        if b['masks'].numel() > 0 and b['masks'].shape[0] > 0:
            max_masks = max(max_masks, b['masks'].shape[0])
    max_masks = max(max_masks, 1)  # At least 1 for padding

    # Get mask dimensions from first sample with valid masks, or use defaults
    mask_h, mask_w = 64, 64
    for b in batch:
        if b['masks'].numel() > 0 and len(b['masks'].shape) >= 2:
            mask_h = b['masks'].shape[-2]
            mask_w = b['masks'].shape[-1]
            break

    masks_list = []
    mask_indices_list = []
    has_mask_list = []

    for b in batch:
        masks_tensor = b['masks']
        indices_tensor = b['mask_indices']

        # Check if this sample has valid masks
        has_valid = masks_tensor.numel() > 0 and masks_tensor.shape[0] > 0
        num_masks = masks_tensor.shape[0] if has_valid else 0

        if has_valid:
            # Has valid masks - pad to max_masks
            padded_masks = torch.zeros(max_masks, mask_h, mask_w, dtype=masks_tensor.dtype)
            padded_masks[:num_masks] = masks_tensor
            masks_list.append(padded_masks)

            # Pad mask indices
            padded_indices = torch.full((max_masks,), -1, dtype=torch.long)
            padded_indices[:num_masks] = indices_tensor[:num_masks]
            mask_indices_list.append(padded_indices)

            has_mask_list.append(True)
        else:
            # No valid masks - create dummy placeholders
            masks_list.append(torch.zeros(max_masks, mask_h, mask_w, dtype=torch.long))
            mask_indices_list.append(torch.full((max_masks,), -1, dtype=torch.long))
            has_mask_list.append(False)

    masks = torch.stack(masks_list)
    mask_indices = torch.stack(mask_indices_list)
    has_mask = torch.tensor(has_mask_list, dtype=torch.bool)

    # Copy metadata
    meta = [b['meta'] for b in batch]

    return {
        'video': videos,
        'labels': labels,
        'masks': masks,
        'mask_indices': mask_indices,
        'has_mask': has_mask,  # Boolean tensor indicating which samples have valid masks
        'meta': meta,
    }


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_entropy(attention: torch.Tensor) -> float:
    """
    Compute normalized entropy of attention distribution.

    Args:
        attention: [batch, num_patches] or [num_patches] attention weights

    Returns:
        Normalized entropy (0 = focused, 1 = uniform)
    """
    if attention.dim() > 1:
        attention = attention.mean(dim=0)

    attention = attention.flatten()
    attention = attention / (attention.sum() + 1e-8)

    entropy = -torch.sum(attention * torch.log(attention + 1e-8))
    max_entropy = torch.log(torch.tensor(len(attention), dtype=torch.float32))

    return (entropy / max_entropy).item()


def create_attention_target(
    mask: torch.Tensor,
    num_patches: int = 256,
    patch_size: int = 16,
    min_coverage: float = 0.01,
) -> Tuple[torch.Tensor, bool]:
    """
    Convert segmentation mask to target attention distribution.

    Attention should focus on non-background regions (classes 1-4).

    Args:
        mask: [batch, H, W] or [batch, 1, H, W] segmentation mask
        num_patches: Number of spatial patches (16x16 = 256)
        patch_size: Grid size (16 for 256 patches)
        min_coverage: Minimum mask coverage to apply attention loss

    Returns:
        attention_target: [batch, num_patches] target distribution
        valid: Whether mask has enough coverage
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    batch_size = mask.shape[0]

    # Resize mask to patch grid
    mask_resized = F.interpolate(
        mask.float(),
        size=(patch_size, patch_size),
        mode='nearest'
    )

    # Create binary: 1 where anatomy is (classes 1-4), 0 for background (class 0)
    anatomy_mask = (mask_resized > 0).float()

    # Flatten to [batch, num_patches]
    anatomy_flat = anatomy_mask.view(batch_size, -1)

    # Check coverage
    coverage = anatomy_flat.sum(dim=1) / num_patches
    valid = coverage >= min_coverage

    # Normalize to probability distribution
    attention_target = anatomy_flat / (anatomy_flat.sum(dim=1, keepdim=True) + 1e-8)

    # For samples with no/little anatomy, use uniform (will be skipped in loss)
    uniform = torch.ones_like(attention_target) / num_patches
    attention_target = torch.where(
        valid.unsqueeze(1).expand_as(attention_target),
        attention_target,
        uniform
    )

    return attention_target, valid


def attention_supervision_loss(
    attention_weights: torch.Tensor,
    attention_target: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_type: str = "kl",
    entropy_target: float = 0.7,
) -> torch.Tensor:
    """
    Compute loss to make attention focus on anatomy regions.

    Args:
        attention_weights: [batch, heads, tokens, tokens] from V-JEPA
        attention_target: [batch, num_patches] from masks
        valid_mask: [batch] boolean mask for valid samples
        loss_type: "kl", "mse", "focal", or "entropy_penalty"
        entropy_target: Target entropy for entropy_penalty loss

    Returns:
        Attention supervision loss (scalar)
    """
    batch_size = attention_weights.shape[0]
    num_heads = attention_weights.shape[1]

    # V-JEPA attention shape: [batch, heads, (temporal*spatial), (temporal*spatial)]
    # We need to extract spatial attention pattern

    # Average over heads
    attn = attention_weights.mean(dim=1)  # [batch, tokens, tokens]

    # Get attention from CLS token (if exists) or average over queries
    # V-JEPA uses [temporal_tokens * spatial_tokens] format
    # Spatial tokens = 256 (16x16), temporal = 8 for 16 frames

    # Take mean attention across all query tokens to all key tokens
    # This gives us a general "where does the model look" pattern
    attn_spatial = attn.mean(dim=1)  # [batch, tokens]

    # Take last 256 tokens (spatial for last temporal bin) or average
    num_spatial = 256
    if attn_spatial.shape[1] >= num_spatial:
        # Reshape to [batch, temporal, spatial] and average over temporal
        num_tokens = attn_spatial.shape[1]
        num_temporal = num_tokens // num_spatial

        if num_temporal * num_spatial == num_tokens:
            attn_spatial = attn_spatial.view(batch_size, num_temporal, num_spatial)
            attn_spatial = attn_spatial.mean(dim=1)  # [batch, 256]
        else:
            # Just take last 256
            attn_spatial = attn_spatial[:, -num_spatial:]

    # Normalize to probability distribution
    attn_spatial = attn_spatial / (attn_spatial.sum(dim=1, keepdim=True) + 1e-8)

    # Only compute loss for valid samples (with sufficient mask coverage)
    if not valid_mask.any():
        return torch.tensor(0.0, device=attention_weights.device)

    attn_valid = attn_spatial[valid_mask]
    target_valid = attention_target[valid_mask]

    if loss_type == "kl":
        # KL divergence: D_KL(target || attention)
        # We want attention to match target
        loss = F.kl_div(
            attn_valid.log(),
            target_valid,
            reduction='batchmean'
        )

    elif loss_type == "mse":
        # MSE loss
        loss = F.mse_loss(attn_valid, target_valid)

    elif loss_type == "focal":
        # Focal-style: penalize more when attention misses anatomy
        # Higher weight where target is high (anatomy regions)
        weight = 1 + target_valid * 10
        loss = ((target_valid - attn_valid) ** 2 * weight).mean()

    elif loss_type == "entropy_penalty":
        # Penalize high entropy (uniform attention)
        # Compute per-sample entropy
        entropy = -torch.sum(attn_valid * torch.log(attn_valid + 1e-8), dim=1)
        max_entropy = torch.log(torch.tensor(attn_valid.shape[1], dtype=torch.float32, device=attn_valid.device))
        normalized_entropy = entropy / max_entropy

        # Penalize entropy above target
        loss = F.relu(normalized_entropy - entropy_target).mean()

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


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
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        attn_out, self.attention_weights = self.attention(
            query, x, x, need_weights=True, average_attn_weights=False
        )
        return self.norm(attn_out.squeeze(1))


class LightweightSegDecoder(nn.Module):
    """Lightweight segmentation decoder."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_classes: int = 5,
        output_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_size = output_size
        self.num_classes = num_classes

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

    def forward(
        self,
        features: torch.Tensor,
        frame_indices: torch.Tensor,
        batch_indices: torch.Tensor,
        spatial_size: int = 16,
    ) -> torch.Tensor:
        if len(frame_indices) == 0:
            return torch.zeros(
                0, self.num_classes, self.output_size, self.output_size,
                device=features.device
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
                    device=features.device
                )
                spatial_feats = torch.cat([spatial_feats, padding], dim=0)

            frame_features.append(spatial_feats)

        frame_features = torch.stack(frame_features, dim=0)
        frame_features = self.proj(frame_features)

        N = frame_features.shape[0]
        frame_features = frame_features.view(N, spatial_size, spatial_size, -1)
        frame_features = frame_features.permute(0, 3, 1, 2).contiguous()

        return self.decoder(frame_features)


class AttentionSupervisedModel(nn.Module):
    """
    V-JEPA with LoRA + Attention Supervision.

    Key additions:
    - Extract attention weights during forward pass
    - Compute attention target from masks
    - Add attention supervision loss
    """

    def __init__(self, config: Dict, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        model_cfg = config.get("model", {})
        lora_cfg = config.get("lora", {})

        # Load V-JEPA backbone
        logger.info("Loading V-JEPA backbone...")
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(
            model_cfg.get("name", "facebook/vjepa2-vitl-fpc16-256-ssv2")
        )
        self.backbone = self.backbone.float()

        # Apply LoRA
        self._apply_lora(lora_cfg)

        hidden_dim = model_cfg.get("hidden_dim", 1024)

        # Task heads
        self.pooler = AttentionPooling(
            hidden_dim=hidden_dim,
            num_heads=model_cfg.get("attention_heads", 8),
            dropout=model_cfg.get("attention_dropout", 0.1),
        )

        self.cvs_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, model_cfg.get("cvs_hidden", 512)),
            nn.GELU(),
            nn.Dropout(model_cfg.get("cvs_dropout", 0.5)),
            nn.Linear(model_cfg.get("cvs_hidden", 512), 3),
        )

        self.seg_head = LightweightSegDecoder(
            hidden_dim=hidden_dim,
            num_classes=model_cfg.get("num_seg_classes", 5),
            output_size=model_cfg.get("seg_output_size", 64),
            dropout=model_cfg.get("seg_dropout", 0.1),
        )

        # Attention extraction hooks
        self.attention_weights = None
        self._register_attention_hooks()

    def _apply_lora(self, lora_cfg: Dict):
        """Apply LoRA to backbone."""
        try:
            from peft import LoraConfig, get_peft_model

            # Freeze base model
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Find target modules
            target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
            target_module_names = []

            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Linear):
                    for target in target_modules:
                        if target in name or target.replace("_proj", "") in name:
                            target_module_names.append(name)
                            break

            if target_module_names:
                lora_config = LoraConfig(
                    r=lora_cfg.get("r", 32),
                    lora_alpha=lora_cfg.get("lora_alpha", 64),
                    target_modules=target_module_names,
                    lora_dropout=lora_cfg.get("lora_dropout", 0.1),
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, lora_config)

                trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
                logger.info(f"Applied LoRA: r={lora_cfg.get('r', 32)}, {trainable:,} trainable params")

        except ImportError:
            logger.warning("PEFT not available - using base backbone")

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        self._attention_hook_handles = []

        # Find the last transformer layer
        actual_backbone = self.backbone
        if hasattr(actual_backbone, 'base_model'):
            actual_backbone = actual_backbone.base_model
        if hasattr(actual_backbone, 'model'):
            actual_backbone = actual_backbone.model

        if hasattr(actual_backbone, 'encoder') and hasattr(actual_backbone.encoder, 'layer'):
            last_layer = actual_backbone.encoder.layer[-1]

            # Hook to capture Q and K for computing attention
            self._q_output = None
            self._k_output = None

            def q_hook(module, input, output):
                self._q_output = output.detach()

            def k_hook(module, input, output):
                self._k_output = output.detach()

            if hasattr(last_layer, 'attention'):
                if hasattr(last_layer.attention, 'query'):
                    q_module = last_layer.attention.query
                    if hasattr(q_module, 'base_layer'):
                        q_module = q_module.base_layer
                    self._attention_hook_handles.append(
                        q_module.register_forward_hook(q_hook)
                    )

                if hasattr(last_layer.attention, 'key'):
                    k_module = last_layer.attention.key
                    if hasattr(k_module, 'base_layer'):
                        k_module = k_module.base_layer
                    self._attention_hook_handles.append(
                        k_module.register_forward_hook(k_hook)
                    )

    def _compute_attention_from_qk(self) -> Optional[torch.Tensor]:
        """Compute attention weights from captured Q and K."""
        if self._q_output is None or self._k_output is None:
            return None

        Q = self._q_output
        K = self._k_output

        B, seq_len, hidden_dim = Q.shape
        num_heads = 16  # ViT-L
        head_dim = hidden_dim // num_heads

        # Reshape to multi-head format
        Q = Q.view(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        K = K.view(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # Compute attention
        scale = head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)

        return attn

    def process_videos(self, videos: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Process videos for V-JEPA input."""
        if videos.dim() == 5:
            B, T, H, W, C = videos.shape
            if C == 3:
                videos = videos.permute(0, 1, 4, 2, 3)

        videos = videos.float() / 255.0 if videos.max() > 1 else videos.float()
        videos = videos.to(device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

        return (videos - mean) / std

    def forward(
        self,
        pixel_values: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
        batch_indices: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional attention extraction.

        Args:
            pixel_values: [B, T, C, H, W] video tensor
            frame_indices: Frame indices for segmentation
            batch_indices: Batch indices for segmentation
            return_attention: Whether to compute and return attention

        Returns:
            Dict with cvs_logits, seg_logits (if applicable), attention_weights (if requested)
        """
        # Get features
        features = self.backbone.get_vision_features(pixel_values_videos=pixel_values)
        features = features.float()

        # CVS classification
        pooled = self.pooler(features)
        cvs_logits = self.cvs_head(pooled)

        result = {
            "cvs_logits": cvs_logits,
            "features": features,
        }

        # Segmentation
        if frame_indices is not None and batch_indices is not None and len(frame_indices) > 0:
            seg_logits = self.seg_head(features, frame_indices, batch_indices)
            result["seg_logits"] = seg_logits

        # Attention weights (computed from hooks)
        if return_attention:
            attention = self._compute_attention_from_qk()
            result["attention_weights"] = attention

        return result


def train_epoch(
    model: AttentionSupervisedModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    config: Dict,
    epoch: int,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch with attention supervision."""
    model.train()

    loss_cfg = config.get("loss", {})
    attn_cfg = config.get("attention_supervision", {})
    train_cfg = config.get("training", {})

    cvs_weight = loss_cfg.get("cvs_weight", 1.0)
    seg_weight = loss_cfg.get("seg_weight", 0.3)
    attn_weight = loss_cfg.get("attention_weight", 0.1)

    warmup_epochs = attn_cfg.get("warmup_epochs", 1)
    apply_attn_loss = attn_cfg.get("enabled", True) and epoch >= warmup_epochs
    attn_loss_type = attn_cfg.get("loss_type", "kl")
    min_mask_coverage = attn_cfg.get("min_mask_coverage", 0.01)

    grad_accum = train_cfg.get("gradient_accumulation", 1)

    # Loss functions
    pos_weight = torch.tensor(
        loss_cfg.get("cvs_pos_weight", [1.0, 3.0, 1.0]),
        device=device
    )
    cvs_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    seg_class_weights = torch.tensor(
        loss_cfg.get("seg_class_weights", [0.1, 5.0, 3.0, 2.0, 2.0]),
        device=device
    )
    num_seg_classes = len(seg_class_weights)  # 5 classes: 0-4
    seg_criterion = nn.CrossEntropyLoss(weight=seg_class_weights, ignore_index=255)

    # Metrics
    total_loss = 0.0
    total_cvs_loss = 0.0
    total_seg_loss = 0.0
    total_attn_loss = 0.0
    total_entropy = 0.0
    num_batches = 0
    num_entropy_samples = 0

    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        videos = batch["video"].to(device)
        labels = batch["labels"].float().to(device)
        masks = batch["masks"].to(device) if "masks" in batch else None
        mask_indices = batch["mask_indices"]
        has_mask = batch["has_mask"].to(device) if "has_mask" in batch else None

        # Process videos
        pixel_values = model.process_videos(videos, device)

        # Prepare segmentation targets (only for samples WITH masks)
        frame_indices = []
        batch_indices = []
        seg_targets = []

        for b_idx in range(len(mask_indices)):
            # Skip samples without valid masks
            if has_mask is not None and not has_mask[b_idx]:
                continue
            for m_idx, frame_idx in enumerate(mask_indices[b_idx]):
                if frame_idx >= 0:
                    frame_indices.append(frame_idx)
                    batch_indices.append(b_idx)
                    seg_targets.append(masks[b_idx, m_idx])

        if frame_indices:
            frame_indices = torch.tensor(frame_indices, device=device)
            batch_indices = torch.tensor(batch_indices, device=device)
            seg_targets = torch.stack(seg_targets).long()
        else:
            frame_indices = torch.tensor([], dtype=torch.long, device=device)
            batch_indices = torch.tensor([], dtype=torch.long, device=device)
            seg_targets = None

        # Forward pass
        with autocast():
            outputs = model(
                pixel_values,
                frame_indices,
                batch_indices,
                return_attention=apply_attn_loss,
            )

            # CVS loss
            cvs_loss = cvs_criterion(outputs["cvs_logits"], labels)

            # Segmentation loss
            seg_loss = torch.tensor(0.0, device=device)
            if "seg_logits" in outputs and seg_targets is not None:
                # Fix mask values: set out-of-range class indices to ignore_index (255)
                seg_targets_fixed = seg_targets.clone().to(device)
                seg_targets_fixed[seg_targets_fixed >= num_seg_classes] = 255
                seg_targets_fixed[seg_targets_fixed < 0] = 255

                seg_loss = seg_criterion(
                    outputs["seg_logits"],
                    seg_targets_fixed
                )

            # Attention supervision loss (only for samples WITH masks)
            attn_loss = torch.tensor(0.0, device=device)
            if apply_attn_loss and outputs.get("attention_weights") is not None:
                # Only compute attention loss for samples that have valid masks
                if masks is not None and has_mask is not None and has_mask.any():
                    # Use first mask (middle frame) for attention target
                    middle_mask = masks[:, 0] if masks.dim() == 4 else masks

                    # Create attention target
                    attention_target, valid_coverage = create_attention_target(
                        middle_mask,
                        min_coverage=min_mask_coverage,
                    )

                    # Combine has_mask with coverage check
                    valid_mask = has_mask & valid_coverage.to(device)

                    if valid_mask.any():
                        attn_loss = attention_supervision_loss(
                            outputs["attention_weights"],
                            attention_target.to(device),
                            valid_mask,
                            loss_type=attn_loss_type,
                        )

            # Total loss
            loss = cvs_weight * cvs_loss + seg_weight * seg_loss + attn_weight * attn_loss
            loss = loss / grad_accum

        # Backward pass
        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Compute attention entropy
        if outputs.get("attention_weights") is not None:
            attn = outputs["attention_weights"]
            # Average over batch, heads, and query dimension
            attn_flat = attn.mean(dim=(0, 1, 2))
            entropy = compute_entropy(attn_flat)
            total_entropy += entropy
            num_entropy_samples += 1

        # Accumulate metrics
        total_loss += loss.item() * grad_accum
        total_cvs_loss += cvs_loss.item()
        total_seg_loss += seg_loss.item()
        total_attn_loss += attn_loss.item()
        num_batches += 1

        # Accumulate predictions
        with torch.no_grad():
            preds = torch.sigmoid(outputs["cvs_logits"]).cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum:.4f}",
            "cvs": f"{cvs_loss.item():.4f}",
            "attn": f"{attn_loss.item():.4f}",
            "ent": f"{total_entropy / max(num_entropy_samples, 1) * 100:.1f}%",
        })

    # Compute mAP
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    from sklearn.metrics import average_precision_score

    aps = []
    for i in range(3):
        if all_labels[:, i].sum() > 0:
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            aps.append(ap)

    mAP = np.mean(aps) if aps else 0.0

    avg_entropy = total_entropy / max(num_entropy_samples, 1) * 100

    return {
        "loss": total_loss / num_batches,
        "cvs_loss": total_cvs_loss / num_batches,
        "seg_loss": total_seg_loss / num_batches,
        "attn_loss": total_attn_loss / num_batches,
        "mAP": mAP,
        "attention_entropy": avg_entropy,
    }


@torch.no_grad()
def validate(
    model: AttentionSupervisedModel,
    dataloader: DataLoader,
    config: Dict,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    loss_cfg = config.get("loss", {})

    pos_weight = torch.tensor(
        loss_cfg.get("cvs_pos_weight", [1.0, 3.0, 1.0]),
        device=device
    )
    cvs_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss = 0.0
    total_entropy = 0.0
    num_batches = 0
    num_entropy_samples = 0

    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Validating"):
        videos = batch["video"].to(device)
        labels = batch["labels"].float().to(device)

        pixel_values = model.process_videos(videos, device)

        outputs = model(pixel_values, return_attention=True)

        cvs_loss = cvs_criterion(outputs["cvs_logits"], labels)

        total_loss += cvs_loss.item()
        num_batches += 1

        # Compute entropy
        if outputs.get("attention_weights") is not None:
            attn = outputs["attention_weights"]
            attn_flat = attn.mean(dim=(0, 1, 2))
            entropy = compute_entropy(attn_flat)
            total_entropy += entropy
            num_entropy_samples += 1

        preds = torch.sigmoid(outputs["cvs_logits"]).cpu()
        all_preds.append(preds)
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    from sklearn.metrics import average_precision_score

    aps = []
    ap_per_class = {}
    for i, name in enumerate(["C1", "C2", "C3"]):
        if all_labels[:, i].sum() > 0:
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            aps.append(ap)
            ap_per_class[name] = ap

    mAP = np.mean(aps) if aps else 0.0
    avg_entropy = total_entropy / max(num_entropy_samples, 1) * 100

    return {
        "loss": total_loss / num_batches,
        "mAP": mAP,
        "C1_AP": ap_per_class.get("C1", 0),
        "C2_AP": ap_per_class.get("C2", 0),
        "C3_AP": ap_per_class.get("C3", 0),
        "attention_entropy": avg_entropy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp11_attention_supervised.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    data_cfg = config.get("data", {})
    results_dir = Path(data_cfg.get("results_dir", "results/exp11_attention_supervised"))
    run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {run_dir}")

    # Create model
    model = AttentionSupervisedModel(config, device)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create datasets
    dataset_cfg = config.get("dataset", {})

    train_dataset = MultiTaskCVSDataset(
        root_dir=data_cfg.get("endoscapes_root"),
        split="train",
        num_frames=dataset_cfg.get("num_frames", 16),
        resolution=dataset_cfg.get("resolution", 256),
        mask_resolution=dataset_cfg.get("mask_resolution", 64),
        augment=True,
        use_synthetic_masks=dataset_cfg.get("use_synthetic_masks", True),
    )

    val_dataset = MultiTaskCVSDataset(
        root_dir=data_cfg.get("endoscapes_root"),
        split="val",
        num_frames=dataset_cfg.get("num_frames", 16),
        resolution=dataset_cfg.get("resolution", 256),
        mask_resolution=dataset_cfg.get("mask_resolution", 64),
        augment=False,
        use_synthetic_masks=dataset_cfg.get("use_synthetic_masks", True),
    )

    train_cfg = config.get("training", {})

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,  # Handle variable mask sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=True,
        collate_fn=custom_collate_fn,  # Handle variable mask sizes
    )

    # Create optimizer with separate LR for LoRA and heads
    lora_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora" in name.lower() or "backbone" in name.lower():
                lora_params.append(param)
            else:
                head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": train_cfg.get("lora_lr", 1e-4)},
        {"params": head_params, "lr": train_cfg.get("head_lr", 5e-4)},
    ], weight_decay=train_cfg.get("weight_decay", 0.05))

    # Create scheduler
    num_epochs = train_cfg.get("epochs", 10)
    total_steps = len(train_loader) * num_epochs // train_cfg.get("gradient_accumulation", 1)
    warmup_steps = int(total_steps * train_cfg.get("warmup_epochs", 1) / num_epochs)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=train_cfg.get("min_lr", 1e-7),
    )
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], [warmup_steps])

    scaler = GradScaler()

    # Training loop
    best_mAP = 0.0
    patience = train_cfg.get("early_stopping_patience", 5)
    patience_counter = 0

    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info(f"Attention supervision: {config.get('attention_supervision', {}).get('enabled', True)}")
    logger.info(f"Attention loss weight: {config.get('loss', {}).get('attention_weight', 0.1)}")
    logger.info("=" * 70)

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, epoch, device
        )

        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"mAP: {train_metrics['mAP']*100:.2f}%, "
            f"Attn Loss: {train_metrics['attn_loss']:.4f}, "
            f"Entropy: {train_metrics['attention_entropy']:.1f}%"
        )

        # Validate
        val_metrics = validate(model, val_loader, config, device)

        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"mAP: {val_metrics['mAP']*100:.2f}%, "
            f"C1: {val_metrics['C1_AP']*100:.2f}%, "
            f"C2: {val_metrics['C2_AP']*100:.2f}%, "
            f"C3: {val_metrics['C3_AP']*100:.2f}%, "
            f"Entropy: {val_metrics['attention_entropy']:.1f}%"
        )

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_mAP,
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        torch.save(checkpoint, run_dir / f"epoch_{epoch}.pt")

        # Check for improvement
        if val_metrics["mAP"] > best_mAP:
            best_mAP = val_metrics["mAP"]
            patience_counter = 0
            torch.save(checkpoint, run_dir / "best_model.pt")
            logger.info(f"New best mAP: {best_mAP*100:.2f}%")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        # Clear cache
        torch.cuda.empty_cache()

    logger.info("=" * 70)
    logger.info(f"Training complete. Best mAP: {best_mAP*100:.2f}%")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
