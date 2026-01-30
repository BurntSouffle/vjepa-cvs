"""
Loss functions for V-JEPA CVS classification.

Includes:
- BCEWithLogitsLoss (standard)
- Focal Loss (for class imbalance)
- Asymmetric Loss (alternative for multi-label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary/multi-label classification.

    Focal loss addresses class imbalance by down-weighting easy examples
    and focusing on hard negatives.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    For multi-label classification (like CVS with 3 independent binary labels),
    this is applied independently to each class.

    Args:
        alpha: Weighting factor for positive class. Can be:
            - float: Same alpha for all classes
            - List[float]: Per-class alpha weights [alpha_C1, alpha_C2, alpha_C3]
            Default: 0.25 (common choice, gives more weight to negatives)
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
            - gamma=0: Equivalent to BCE
            - gamma=2: Common choice (original paper)
            Default: 2.0
        reduction: 'mean', 'sum', or 'none'

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
        https://arxiv.org/abs/1708.02002

    Example:
        >>> criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 3)  # batch=32, 3 classes
        >>> labels = torch.randint(0, 2, (32, 3)).float()
        >>> loss = criterion(logits, labels)
    """

    def __init__(
        self,
        alpha: Union[float, List[float]] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Handle alpha (can be scalar or per-class)
        if isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha))
        else:
            self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (before sigmoid), shape (B, C)
            targets: Binary labels, shape (B, C)

        Returns:
            Focal loss value
        """
        # Get probabilities
        probs = torch.sigmoid(logits)

        # Compute BCE component (more numerically stable than manual log)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Compute p_t (probability of correct class)
        # p_t = p if y=1, else (1-p)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha weight
        if isinstance(self.alpha, torch.Tensor):
            # Per-class alpha: alpha for positives, (1-alpha) for negatives
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            # Scalar alpha
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combine: focal_loss = -alpha_t * (1-p_t)^gamma * log(p_t)
        # Since BCE = -log(p_t), we have: focal_loss = alpha_t * focal_weight * BCE
        focal_loss = alpha_t * focal_weight * bce_loss

        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossWithPosWeight(nn.Module):
    """
    Focal Loss with additional positive class weighting.

    Combines focal loss focusing with pos_weight for handling severe imbalance.
    Useful when positive samples are very rare (like CVS criteria).

    Args:
        alpha: Focal loss alpha (balancing factor)
        gamma: Focal loss gamma (focusing parameter)
        pos_weight: Additional weight for positive samples per class
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: Union[float, List[float]] = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[List[float]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha))
        else:
            self.alpha = alpha

        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # BCE with optional pos_weight
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction="none"
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Variant of focal loss that treats positive and negative samples asymmetrically.
    Often works better than focal loss for multi-label problems.

    Args:
        gamma_neg: Focusing parameter for negative samples (typically 4)
        gamma_pos: Focusing parameter for positive samples (typically 1)
        clip: Probability clipping for numerical stability
        reduction: 'mean', 'sum', or 'none'

    Reference:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification", 2021
        https://arxiv.org/abs/2009.14119
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Asymmetric clipping for negatives
        probs_neg = probs.clamp(min=self.clip)

        # Separate positive and negative losses
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log(1 - probs_neg)

        # Asymmetric focusing
        if self.gamma_pos > 0:
            loss_pos = loss_pos * ((1 - probs) ** self.gamma_pos)
        if self.gamma_neg > 0:
            loss_neg = loss_neg * (probs_neg ** self.gamma_neg)

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def create_loss_function(config: dict, device: torch.device) -> nn.Module:
    """
    Factory function to create loss function from config.

    Config options:
        loss:
          type: "bce_with_logits" | "focal" | "focal_with_pos_weight" | "asymmetric"
          use_class_weights: false
          pos_weight: [1.0, 1.0, 1.0]

          # Focal loss specific
          focal_alpha: 0.25        # or [0.25, 0.25, 0.25] for per-class
          focal_gamma: 2.0

          # Asymmetric loss specific
          gamma_neg: 4.0
          gamma_pos: 1.0
          clip: 0.05

    Returns:
        Loss function module
    """
    loss_config = config["loss"]
    loss_type = loss_config.get("type", "bce_with_logits")

    if loss_type == "bce_with_logits":
        # Standard BCE
        if loss_config.get("use_class_weights", False):
            pos_weight = torch.tensor(loss_config["pos_weight"]).to(device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCEWithLogitsLoss()

    elif loss_type == "focal":
        # Basic focal loss
        alpha = loss_config.get("focal_alpha", 0.25)
        gamma = loss_config.get("focal_gamma", 2.0)
        return BinaryFocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == "focal_with_pos_weight":
        # Focal loss + pos_weight
        alpha = loss_config.get("focal_alpha", 0.25)
        gamma = loss_config.get("focal_gamma", 2.0)
        pos_weight = loss_config.get("pos_weight", None)
        return FocalLossWithPosWeight(alpha=alpha, gamma=gamma, pos_weight=pos_weight)

    elif loss_type == "asymmetric":
        # Asymmetric loss
        gamma_neg = loss_config.get("gamma_neg", 4.0)
        gamma_pos = loss_config.get("gamma_pos", 1.0)
        clip = loss_config.get("clip", 0.05)
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Quick test
if __name__ == "__main__":
    print("Testing loss functions...")

    # Test data
    batch_size = 8
    num_classes = 3
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Targets:\n{targets}")

    # Test BCE
    bce = nn.BCEWithLogitsLoss()
    bce_loss = bce(logits, targets)
    print(f"\nBCE Loss: {bce_loss.item():.4f}")

    # Test Focal Loss (various gamma)
    for gamma in [0.0, 1.0, 2.0, 5.0]:
        focal = BinaryFocalLoss(alpha=0.25, gamma=gamma)
        focal_loss = focal(logits, targets)
        print(f"Focal Loss (gamma={gamma}): {focal_loss.item():.4f}")

    # Test per-class alpha
    focal_perclass = BinaryFocalLoss(alpha=[0.3, 0.5, 0.7], gamma=2.0)
    focal_perclass_loss = focal_perclass(logits, targets)
    print(f"Focal Loss (per-class alpha): {focal_perclass_loss.item():.4f}")

    # Test Focal with pos_weight
    focal_pw = FocalLossWithPosWeight(alpha=0.25, gamma=2.0, pos_weight=[2.0, 3.0, 4.0])
    focal_pw_loss = focal_pw(logits, targets)
    print(f"Focal+PosWeight Loss: {focal_pw_loss.item():.4f}")

    # Test Asymmetric Loss
    asym = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
    asym_loss = asym(logits, targets)
    print(f"Asymmetric Loss: {asym_loss.item():.4f}")

    # Test factory function
    print("\n--- Testing factory function ---")
    test_configs = [
        {"loss": {"type": "bce_with_logits"}},
        {"loss": {"type": "focal", "focal_alpha": 0.25, "focal_gamma": 2.0}},
        {"loss": {"type": "focal_with_pos_weight", "focal_alpha": 0.5, "focal_gamma": 2.0, "pos_weight": [2.0, 2.0, 2.0]}},
        {"loss": {"type": "asymmetric", "gamma_neg": 4.0, "gamma_pos": 1.0}},
    ]

    for cfg in test_configs:
        loss_fn = create_loss_function(cfg, torch.device("cpu"))
        loss_val = loss_fn(logits, targets)
        print(f"{cfg['loss']['type']}: {loss_val.item():.4f}")

    print("\nAll tests passed!")
