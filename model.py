"""
V-JEPA 2 CVS Classification Model.

Supports multiple pooling strategies and head architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over token dimension.

    Learns a query vector that attends to all tokens,
    producing a weighted sum as the pooled representation.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Learnable query token
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm for output
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_tokens, hidden_dim) token features

        Returns:
            pooled: (B, hidden_dim) pooled representation
        """
        B = x.size(0)

        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # (B, 1, hidden_dim)

        # Attend to all tokens
        attn_out, _ = self.attention(query, x, x)  # (B, 1, hidden_dim)

        # Remove sequence dimension and normalize
        pooled = self.norm(attn_out.squeeze(1))  # (B, hidden_dim)

        return pooled


class MeanPooling(nn.Module):
    """Simple mean pooling over token dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_tokens, hidden_dim) token features

        Returns:
            pooled: (B, hidden_dim) mean-pooled representation
        """
        return x.mean(dim=1)


class MLPHead(nn.Module):
    """Original MLP classification head."""

    def __init__(
        self,
        hidden_dim: int,
        classifier_hidden: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SimpleHead(nn.Module):
    """
    Simple classification head with minimal layers.

    Architecture: LayerNorm -> Dropout -> Linear

    Purpose: Reduce overfitting by minimizing learnable parameters.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class VJEPA_CVS(nn.Module):
    """
    V-JEPA 2 based CVS classifier.

    Architecture:
        V-JEPA Encoder (frozen or partially trainable)
            -> Pooling (mean or attention)
            -> Head (MLP or simple)
            -> 3 logits (C1, C2, C3)

    Configurable components:
        - pooling_type: "mean" or "attention"
        - head_type: "mlp" or "simple"
        - freeze_backbone: full freeze or partial unfreeze
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        hidden_dim: int = 1024,
        classifier_hidden: int = 512,
        num_classes: int = 3,
        dropout: float = 0.3,
        pooling_type: str = "mean",
        head_type: str = "mlp",
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
    ):
        """
        Args:
            model_name: HuggingFace model name for V-JEPA 2
            freeze_backbone: Whether to freeze the V-JEPA encoder
            unfreeze_last_n_layers: Number of last transformer layers to unfreeze (0 = all frozen)
            hidden_dim: V-JEPA hidden dimension (1024 for ViT-L)
            classifier_hidden: Hidden dimension of classifier MLP
            num_classes: Number of output classes (3 for CVS)
            dropout: Dropout rate in classifier
            pooling_type: "mean" or "attention"
            head_type: "mlp" or "simple"
            attention_heads: Number of attention heads (if using attention pooling)
            attention_dropout: Dropout for attention pooling
        """
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.head_type = head_type

        # Load V-JEPA model
        print(f"Loading V-JEPA model: {model_name}")
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

        # Note: We skip AutoVideoProcessor - it's extremely slow (18s/batch)
        # process_videos() does fast manual preprocessing instead (1.3s/batch, 14x faster)

        # Setup backbone freezing (full or partial)
        self._setup_backbone_freezing(freeze_backbone, unfreeze_last_n_layers)

        # Pooling layer
        if pooling_type == "attention":
            print(f"Using attention pooling with {attention_heads} heads")
            self.pooler = AttentionPooling(
                hidden_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
            )
        else:
            print("Using mean pooling")
            self.pooler = MeanPooling()

        # Classification head
        if head_type == "simple":
            print("Using simple head (LayerNorm -> Dropout -> Linear)")
            self.classifier = SimpleHead(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            print(f"Using MLP head (hidden={classifier_hidden})")
            self.classifier = MLPHead(
                hidden_dim=hidden_dim,
                classifier_hidden=classifier_hidden,
                num_classes=num_classes,
                dropout=dropout,
            )

    def _setup_backbone_freezing(self, freeze_backbone: bool, unfreeze_last_n_layers: int = 0):
        """
        Setup backbone freezing with optional partial unfreezing of last N layers.

        V-JEPA ViT-L has 24 transformer layers. This allows fine-tuning only the
        last N layers while keeping early layers (generic features) frozen.

        Args:
            freeze_backbone: If True and unfreeze_last_n_layers=0, freeze everything
            unfreeze_last_n_layers: Number of last layers to unfreeze (0 = all frozen)
        """
        # First, freeze ALL backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Count total parameters for reporting
        total_backbone_params = sum(p.numel() for p in self.backbone.parameters())

        if freeze_backbone and unfreeze_last_n_layers == 0:
            # Full freeze
            print("Freezing entire V-JEPA backbone")
            self.backbone.eval()
            frozen_params = total_backbone_params
            unfrozen_params = 0
        elif unfreeze_last_n_layers > 0:
            # Partial unfreeze - unfreeze last N transformer layers
            total_layers = 24  # ViT-L has 24 layers
            start_unfreeze = total_layers - unfreeze_last_n_layers

            print(f"Unfreezing last {unfreeze_last_n_layers} transformer layers (layers {start_unfreeze}-{total_layers-1})")

            unfrozen_params = 0
            for name, param in self.backbone.named_parameters():
                # Check if this param belongs to a layer >= start_unfreeze
                # V-JEPA uses 'encoder.layer.X.' pattern (note: singular 'layer', not 'layers')
                should_unfreeze = False

                # Try different naming patterns used by different model architectures
                for pattern in ['encoder.layer.', 'encoder.layers.', 'blocks.', 'layer.']:
                    if pattern in name:
                        # Extract layer number
                        try:
                            # Find the number after the pattern
                            after_pattern = name.split(pattern)[1]
                            layer_num = int(after_pattern.split('.')[0])
                            if layer_num >= start_unfreeze:
                                should_unfreeze = True
                                break
                        except (ValueError, IndexError):
                            continue

                # Also unfreeze final layer norm if it exists (encoder.norm)
                if name == 'encoder.norm.weight' or name == 'encoder.norm.bias':
                    should_unfreeze = True

                if should_unfreeze:
                    param.requires_grad = True
                    unfrozen_params += param.numel()

            frozen_params = total_backbone_params - unfrozen_params

            # CRITICAL: Convert entire backbone to float32 for stable training
            # fp16 parameters cause NaN during optimizer updates
            # This uses more memory but is necessary for fine-tuning
            self.backbone = self.backbone.float()
            print("Converted backbone to float32 for stable fine-tuning")

            # Keep backbone in train mode for unfrozen layers
            # But we'll set it to eval in forward if fully frozen
        else:
            # Full unfreeze (freeze_backbone=False and unfreeze_last_n_layers=0)
            print("Unfreezing entire V-JEPA backbone (full fine-tuning)")
            for param in self.backbone.parameters():
                param.requires_grad = True
            frozen_params = 0
            unfrozen_params = total_backbone_params

        # Report parameter counts
        print(f"Backbone total params: {total_backbone_params / 1e6:.1f}M")
        print(f"Backbone frozen params: {frozen_params / 1e6:.1f}M")
        print(f"Backbone unfrozen params: {unfrozen_params / 1e6:.1f}M")

        # Store for later use
        self._backbone_frozen_params = frozen_params
        self._backbone_unfrozen_params = unfrozen_params

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pixel_values_videos: Processed video tensor from processor
                Shape: (B, T, C, H, W)

        Returns:
            logits: (B, num_classes) classification logits
        """
        # Get V-JEPA features
        # Use no_grad only if backbone is fully frozen (no unfrozen layers)
        if self.freeze_backbone and self.unfreeze_last_n_layers == 0:
            with torch.no_grad():
                features = self.backbone.get_vision_features(
                    pixel_values_videos=pixel_values_videos
                )
        else:
            features = self.backbone.get_vision_features(
                pixel_values_videos=pixel_values_videos
            )

        # features shape: (B, num_tokens, hidden_dim)
        # Convert to float32 for classifier (backbone outputs fp16)
        features = features.float()

        # Pool over tokens
        pooled = self.pooler(features)  # (B, hidden_dim)

        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)

        return logits

    def process_videos(self, videos: list, device: torch.device) -> torch.Tensor:
        """
        Process raw videos - FAST version bypassing slow HuggingFace processor.

        The HuggingFace AutoVideoProcessor takes ~18s per batch of 32 videos.
        This manual implementation takes ~1.3s (14x faster).

        Args:
            videos: List of numpy arrays (T, H, W, C) uint8
            device: Target device

        Returns:
            Processed tensor ready for forward pass (B, T, C, H, W)
        """
        import numpy as np

        # Stack into tensor: (B, T, H, W, C)
        video_tensor = torch.from_numpy(np.stack(videos))

        # Permute to (B, T, C, H, W) - V-JEPA expected format
        video_tensor = video_tensor.permute(0, 1, 4, 2, 3).contiguous()

        # Move to device and convert to float [0, 1]
        video_tensor = video_tensor.to(device).float() / 255.0

        # Apply ImageNet normalization (same as V-JEPA processor)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        video_tensor = (video_tensor - mean) / std

        return video_tensor

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(config: dict) -> VJEPA_CVS:
    """Create model from config."""
    model_cfg = config["model"]

    model = VJEPA_CVS(
        model_name=model_cfg["name"],
        freeze_backbone=model_cfg["freeze_backbone"],
        unfreeze_last_n_layers=model_cfg.get("unfreeze_last_n_layers", 0),
        hidden_dim=model_cfg["hidden_dim"],
        classifier_hidden=model_cfg.get("classifier_hidden", 512),
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        pooling_type=model_cfg.get("pooling_type", "mean"),
        head_type=model_cfg.get("head_type", "mlp"),
        attention_heads=model_cfg.get("attention_heads", 8),
        attention_dropout=model_cfg.get("attention_dropout", 0.1),
    )
    return model


if __name__ == "__main__":
    # Test the model with different configurations
    import numpy as np
    from utils import load_config

    config = load_config("config.yaml")

    print("=" * 60)
    print("Testing model configurations")
    print("=" * 60)

    # Test configurations
    test_configs = [
        {"pooling_type": "mean", "head_type": "mlp"},
        {"pooling_type": "attention", "head_type": "mlp"},
        {"pooling_type": "mean", "head_type": "simple"},
        {"pooling_type": "attention", "head_type": "simple"},
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cfg in test_configs:
        print(f"\n--- Testing: pooling={cfg['pooling_type']}, head={cfg['head_type']} ---")

        # Update config
        config["model"]["pooling_type"] = cfg["pooling_type"]
        config["model"]["head_type"] = cfg["head_type"]

        # Create model
        model = create_model(config)
        model = model.to(device)

        print(f"Total parameters: {model.get_num_total_params() / 1e6:.1f}M")
        print(f"Trainable parameters: {model.get_num_trainable_params() / 1e6:.1f}M")

        # Test with dummy input
        batch_size = 2
        num_frames = config["dataset"]["num_frames"]
        resolution = config["dataset"]["resolution"]

        dummy_videos = [
            np.random.randint(0, 255, (num_frames, resolution, resolution, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

        pixel_values = model.process_videos(dummy_videos, device)
        print(f"Processed shape: {pixel_values.shape}")

        with torch.no_grad():
            logits = model(pixel_values)
            print(f"Output logits shape: {logits.shape}")

        # Clear GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("All configurations tested successfully!")
    print("=" * 60)
