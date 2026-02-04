# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for anatomy-guided masking in surgical video pretraining

import math
from multiprocessing import Value
from logging import getLogger

import torch
import torch.nn.functional as F

_GLOBAL_SEED = 0
logger = getLogger()


class AnatomyGuidedMaskCollator(object):
    """
    Mask collator that biases mask placement toward anatomy regions.

    This extends the original MaskCollator to support anatomy-guided sampling
    where masks are more likely to be placed over anatomical structures
    (cystic plate, calot triangle, vessels) rather than background/tools.

    When anatomy_map is not provided, falls back to random sampling.
    """

    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
        anatomy_bias=0.7,  # Probability of using anatomy-guided placement
    ):
        super(AnatomyGuidedMaskCollator, self).__init__()

        self.anatomy_bias = anatomy_bias
        self.mask_generators = []

        for m in cfgs_mask:
            mask_generator = _AnatomyGuidedMaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=m.get('spatial_scale'),
                temporal_pred_mask_scale=m.get('temporal_scale'),
                aspect_ratio=m.get('aspect_ratio'),
                npred=m.get('num_blocks'),
                max_context_frames_ratio=m.get('max_temporal_keep', 1.0),
                max_keep=m.get('max_keep', None),
                anatomy_bias=anatomy_bias,
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch, anatomy_maps=None):
        """
        Create encoder and predictor masks.

        Args:
            batch: List of video clips
            anatomy_maps: Optional list of anatomy probability maps, one per sample
                         Shape: (batch_size, H_patches, W_patches) or None

        Returns:
            collated_batch: Collated video data
            collated_masks_enc: Encoder masks (which patches to keep)
            collated_masks_pred: Predictor masks (which patches to predict)
        """
        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred, collated_masks_enc = [], []
        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(batch_size, anatomy_maps)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred


class _AnatomyGuidedMaskGenerator(object):
    """
    Internal mask generator that supports anatomy-guided sampling.
    """

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        anatomy_bias=0.7,
    ):
        super(_AnatomyGuidedMaskGenerator, self).__init__()

        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        self.crop_size = crop_size
        self.height = crop_size[0] // spatial_patch_size[0] if isinstance(spatial_patch_size, tuple) else crop_size[0] // spatial_patch_size
        self.width = crop_size[1] // spatial_patch_size[1] if isinstance(spatial_patch_size, tuple) else crop_size[1] // spatial_patch_size
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))
        self.max_keep = max_keep
        self.anatomy_bias = anatomy_bias
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self,
        generator,
        temporal_scale,
        spatial_scale,
        aspect_ratio_scale
    ):
        """Sample block dimensions (same as original)."""
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size, anatomy_map=None):
        """
        Sample a block mask, optionally biased toward anatomy regions.

        Args:
            b_size: (t, h, w) block dimensions
            anatomy_map: Optional (H, W) tensor of anatomy probabilities

        Returns:
            mask: (duration, height, width) binary mask where 0 = masked (to predict)
        """
        t, h, w = b_size

        # Determine if we use anatomy-guided sampling for this block
        use_anatomy = (
            anatomy_map is not None and
            torch.rand(1).item() < self.anatomy_bias
        )

        if use_anatomy:
            # Anatomy-guided: sample position weighted by anatomy probability
            valid_h = self.height - h + 1
            valid_w = self.width - w + 1

            if valid_h <= 0 or valid_w <= 0:
                # Block is too large, fall back to random
                use_anatomy = False
            else:
                # Resize anatomy map to patch grid if needed
                if anatomy_map.shape != (self.height, self.width):
                    anatomy_map = F.interpolate(
                        anatomy_map.unsqueeze(0).unsqueeze(0).float(),
                        size=(self.height, self.width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                # Average pool to get weights for each valid position
                # This computes the average anatomy probability within each possible block placement
                weights = F.avg_pool2d(
                    anatomy_map.unsqueeze(0).unsqueeze(0).float(),
                    kernel_size=(h, w),
                    stride=1,
                    padding=0
                ).squeeze()

                # Ensure we have the right shape
                if weights.dim() == 0:
                    weights = weights.unsqueeze(0).unsqueeze(0)
                elif weights.dim() == 1:
                    weights = weights.unsqueeze(0)

                weights = weights[:valid_h, :valid_w].flatten()

                # Add small epsilon and normalize
                weights = weights + 1e-8
                weights = weights / weights.sum()

                # Sample position from weighted distribution
                idx = torch.multinomial(weights, 1).item()
                top = idx // valid_w
                left = idx % valid_w

        if not use_anatomy:
            # Random placement (original behavior)
            top = torch.randint(0, max(1, self.height - h + 1), (1,)).item()
            left = torch.randint(0, max(1, self.width - w + 1), (1,)).item()

        # Temporal position (always random - we want temporal diversity)
        start = torch.randint(0, max(1, self.duration - t + 1), (1,)).item()

        # Create mask (1 = keep, 0 = predict)
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start:start+t, top:top+h, left:left+w] = 0

        # Context mask will only span the first X frames
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration:, :, :] = 0

        return mask

    def __call__(self, batch_size, anatomy_maps=None):
        """
        Create encoder and predictor masks for a batch.

        Args:
            batch_size: Number of samples
            anatomy_maps: Optional list of anatomy maps, one per sample

        Returns:
            collated_masks_enc: (batch_size, num_enc_patches) indices of patches to encode
            collated_masks_pred: (batch_size, num_pred_patches) indices of patches to predict
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width

        for i in range(batch_size):
            # Get anatomy map for this sample if available
            anatomy_map = None
            if anatomy_maps is not None and i < len(anatomy_maps):
                anatomy_map = anatomy_maps[i]

            empty_context = True
            max_attempts = 10  # Prevent infinite loop
            attempts = 0

            while empty_context and attempts < max_attempts:
                attempts += 1

                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size, anatomy_map)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                # Handle edge cases where squeeze might reduce dimensions too much
                if mask_e.dim() == 0:
                    mask_e = mask_e.unsqueeze(0)
                if mask_p.dim() == 0:
                    mask_p = mask_p.unsqueeze(0)

                empty_context = len(mask_e) == 0

                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred


# Convenience function to convert segmentation mask to anatomy probability map
def segmentation_to_anatomy_map(seg_mask, class_weights=None):
    """
    Convert a segmentation mask to an anatomy probability map.

    Args:
        seg_mask: (H, W) tensor with class indices
        class_weights: Optional dict mapping class_id -> weight (0-1)
                      Default weights favor anatomical structures

    Returns:
        anatomy_map: (H, W) tensor of anatomy probabilities [0, 1]
    """
    if class_weights is None:
        # Default weights for Endoscapes/SAGES classes
        # Higher weight = more likely to be masked (predicted)
        class_weights = {
            0: 0.1,    # background - low priority
            1: 1.0,    # cystic_plate / generic anatomy - HIGH
            2: 1.0,    # calot_triangle - HIGH
            3: 1.0,    # cystic_artery - HIGH
            4: 1.0,    # cystic_duct - HIGH
            5: 0.8,    # gallbladder - high (context for anatomy)
            6: 0.3,    # tool - medium (useful to predict, but not primary)
            255: 0.0,  # ignore
        }

    anatomy_map = torch.zeros_like(seg_mask, dtype=torch.float32)

    for class_id, weight in class_weights.items():
        anatomy_map[seg_mask == class_id] = weight

    return anatomy_map
