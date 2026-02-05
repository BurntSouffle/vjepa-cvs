# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Anatomy-guided V-JEPA pretraining on surgical videos

import os
import sys
import copy
import time
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.surgical_video_dataset import (
    SurgicalVideoDataset,
    make_surgical_videodataset,
    collate_surgical_videos,
)
from src.masks.anatomy_guided_multiblock3d import (
    AnatomyGuidedMaskCollator,
    segmentation_to_anatomy_map,
)
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    AverageMeter,
)
from src.utils.tensors import repeat_interleave_batch

from app.vjepa.utils import (
    load_checkpoint,
    init_video_model,
    init_opt,
)
from app.vjepa.transforms import make_transforms

# Logging configuration
log_timings = True
log_freq = 10
checkpoint_freq = 1

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)


def compute_attention_entropy(attention_weights):
    """
    Compute entropy of attention weights to track focus changes.

    Args:
        attention_weights: (batch, heads, seq_len, seq_len) attention matrices

    Returns:
        entropy: Mean entropy across batch and heads (higher = more uniform)
    """
    # Average across heads
    attn = attention_weights.mean(dim=1)  # (batch, seq_len, seq_len)

    # Compute entropy per query position
    # entropy = -sum(p * log(p))
    eps = 1e-8
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)  # (batch, seq_len)

    # Normalize by max entropy (uniform distribution)
    seq_len = attention_weights.shape[-1]
    max_entropy = math.log(seq_len)
    normalized_entropy = entropy / max_entropy

    return normalized_entropy.mean().item()


def main(args, resume_preempt=False):
    """
    Main training function for anatomy-guided V-JEPA pretraining.
    """

    # ----------------------------------------------------------------------- #
    #  PARSE CONFIG
    # ----------------------------------------------------------------------- #

    # -- META
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    r_file = cfgs_meta.get('read_checkpoint', None)
    seed = cfgs_meta.get('seed', _GLOBAL_SEED)
    use_sdpa = cfgs_meta.get('use_sdpa', True)
    which_dtype = cfgs_meta.get('dtype', 'bfloat16')

    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- DATA
    cfgs_data = args.get('data')
    endoscapes_config = cfgs_data.get('endoscapes', {})
    sages_config = cfgs_data.get('sages', {})
    batch_size = cfgs_data.get('batch_size', 4)
    num_frames = cfgs_data.get('num_frames', 16)
    tubelet_size = cfgs_data.get('tubelet_size', 2)
    crop_size = cfgs_data.get('crop_size', 256)
    centre_crop = cfgs_data.get('centre_crop', 480)
    patch_size = cfgs_data.get('patch_size', 16)
    num_workers = cfgs_data.get('num_workers', 8)
    pin_mem = cfgs_data.get('pin_mem', True)

    # -- MASK
    cfgs_mask = args.get('mask')
    anatomy_bias = cfgs_mask.get('anatomy_bias', 0.7) if isinstance(cfgs_mask, dict) else 0.7
    # Extract mask configs (skip 'type' and 'anatomy_bias' keys)
    mask_configs = [m for m in cfgs_mask if isinstance(m, dict) and 'num_blocks' in m]

    # -- MODEL
    cfgs_model = args.get('model')
    model_name = cfgs_model.get('model_name', 'vit_large')
    pred_depth = cfgs_model.get('pred_depth', 12)
    pred_embed_dim = cfgs_model.get('pred_embed_dim', 384)
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)
    pretrained_checkpoint = cfgs_model.get('pretrained_checkpoint', None)
    load_pretrained = cfgs_model.get('load_pretrained', True)

    # -- OPTIMIZATION
    cfgs_opt = args.get('optimization')
    epochs = cfgs_opt.get('epochs', 20)
    ipe = cfgs_opt.get('ipe', 100)
    ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
    warmup = cfgs_opt.get('warmup', 2)
    start_lr = cfgs_opt.get('start_lr', 2e-5)
    lr = cfgs_opt.get('lr', 6.25e-5)
    final_lr = cfgs_opt.get('final_lr', 1e-7)
    weight_decay = cfgs_opt.get('weight_decay', 0.04)
    final_weight_decay = cfgs_opt.get('final_weight_decay', 0.4)
    clip_grad = cfgs_opt.get('clip_grad', 10.0)
    ema = cfgs_opt.get('ema', [0.998, 1.0])
    gradient_accumulation = args.get('gradient_accumulation', 16)

    # -- LOGGING
    cfgs_logging = args.get('logging')
    log_folder = cfgs_logging.get('folder', './logs')
    tag = cfgs_logging.get('write_tag', 'surgical_pretrain')
    log_attention_every = cfgs_logging.get('log_attention_entropy_every', 500)

    # -- LOSS
    cfgs_loss = args.get('loss')
    loss_exp = cfgs_loss.get('loss_exp', 1.0)
    reg_coeff = cfgs_loss.get('reg_coeff', 0.0)

    # ----------------------------------------------------------------------- #
    #  INITIALIZE DISTRIBUTED
    # ----------------------------------------------------------------------- #
    try:
        world_size, rank = init_distributed()
    except Exception:
        world_size, rank = 1, 0
        logger.info('Running in single-GPU mode')

    logger.info(f'World size: {world_size}, Rank: {rank}')

    # Set seed
    seed = seed + rank
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----------------------------------------------------------------------- #
    #  SETUP LOGGING
    # ----------------------------------------------------------------------- #
    log_folder = Path(log_folder)
    log_folder.mkdir(parents=True, exist_ok=True)

    csv_logger = None
    if rank == 0:
        log_file = log_folder / f'{tag}.csv'
        csv_logger = CSVLogger(
            log_file,
            ('%d', 'epoch'),
            ('%d', 'itr'),
            ('%.5f', 'loss'),
            ('%.2e', 'lr'),
            ('%.4f', 'attention_entropy'),
        )

    # ----------------------------------------------------------------------- #
    #  CREATE DATA LOADER
    # ----------------------------------------------------------------------- #
    logger.info('Creating surgical video dataset...')

    # Create transforms
    transform = make_transforms(
        crop_size=crop_size,
        train=True,
        **args.get('data_aug', {})
    )

    # Create anatomy-guided mask collator
    mask_collator = AnatomyGuidedMaskCollator(
        cfgs_mask=mask_configs,
        crop_size=(crop_size, crop_size),
        num_frames=num_frames,
        patch_size=(patch_size, patch_size),
        tubelet_size=tubelet_size,
        anatomy_bias=anatomy_bias,
    )

    # Create dataset
    dataset, data_loader, dist_sampler = make_surgical_videodataset(
        endoscapes_config=endoscapes_config,
        sages_config=sages_config,
        batch_size=batch_size,
        frames_per_clip=num_frames,
        crop_size=crop_size,
        centre_crop=centre_crop,
        transform=transform,
        rank=rank,
        world_size=world_size,
        collator=None,  # We'll collate manually to handle anatomy maps
        num_workers=num_workers,
        pin_mem=pin_mem,
    )

    logger.info(f'Dataset size: {len(dataset)}')
    logger.info(f'Batches per epoch: {len(data_loader)}')

    # Adjust iterations per epoch
    ipe = len(data_loader) if ipe <= 0 else min(ipe, len(data_loader))
    logger.info(f'Iterations per epoch: {ipe}')

    # ----------------------------------------------------------------------- #
    #  CREATE MODEL
    # ----------------------------------------------------------------------- #
    logger.info('Creating V-JEPA model...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize encoder and predictor
    encoder, predictor = init_video_model(
        model_name=model_name,
        crop_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
    )

    # Create target encoder (EMA)
    target_encoder = copy.deepcopy(encoder)

    # Load pretrained checkpoint if specified
    if load_pretrained and pretrained_checkpoint:
        logger.info(f'Loading pretrained checkpoint: {pretrained_checkpoint}')
        checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
        encoder.load_state_dict(checkpoint.get('encoder', checkpoint), strict=False)
        target_encoder.load_state_dict(checkpoint.get('encoder', checkpoint), strict=False)
        if 'predictor' in checkpoint:
            predictor.load_state_dict(checkpoint['predictor'], strict=False)

    # Move to device
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    target_encoder = target_encoder.to(device)

    # Freeze target encoder
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Wrap with DDP if distributed
    if world_size > 1:
        encoder = DistributedDataParallel(encoder, device_ids=[rank])
        predictor = DistributedDataParallel(predictor, device_ids=[rank])

    logger.info(f'Encoder params: {sum(p.numel() for p in encoder.parameters()):,}')
    logger.info(f'Predictor params: {sum(p.numel() for p in predictor.parameters()):,}')

    # ----------------------------------------------------------------------- #
    #  CREATE OPTIMIZER
    # ----------------------------------------------------------------------- #
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=weight_decay,
        final_wd=final_weight_decay,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
    )

    # ----------------------------------------------------------------------- #
    #  LOAD CHECKPOINT (for resuming)
    # ----------------------------------------------------------------------- #
    start_epoch = 0
    if load_model and r_file:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            r_file=r_file,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )

    # ----------------------------------------------------------------------- #
    #  TRAINING LOOP
    # ----------------------------------------------------------------------- #
    logger.info('Starting training...')

    for epoch in range(start_epoch, epochs):
        logger.info(f'Epoch {epoch+1}/{epochs}')

        if dist_sampler is not None:
            dist_sampler.set_epoch(epoch)

        encoder.train()
        predictor.train()

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, batch_data in enumerate(data_loader):
            if itr >= ipe:
                break

            # Unpack batch
            videos, anatomy_maps, clip_infos = batch_data

            # Move to device
            videos = videos.to(device, non_blocking=True)  # (B, T, C, H, W)

            # Rearrange to (B, C, T, H, W) for V-JEPA
            videos = videos.permute(0, 2, 1, 3, 4)

            # Generate masks with anatomy guidance
            valid_anatomy_maps = [am for am in anatomy_maps if am is not None]
            if valid_anatomy_maps:
                # Pool anatomy maps to patch resolution for mask sampling
                patch_h = crop_size // patch_size
                patch_w = crop_size // patch_size
                pooled_maps = []
                for am in valid_anatomy_maps:
                    am = am.float().unsqueeze(0).unsqueeze(0)
                    am = F.adaptive_avg_pool2d(am, (patch_h, patch_w))
                    pooled_maps.append(am.squeeze())
            else:
                pooled_maps = None

            # Generate masks
            masks_enc, masks_pred = [], []
            for mask_gen in mask_collator.mask_generators:
                m_enc, m_pred = mask_gen(len(videos), pooled_maps)
                masks_enc.append(m_enc.to(device))
                masks_pred.append(m_pred.to(device))

            iter_start = time.time()

            # Forward pass
            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                # Encode context (unmasked patches)
                h = encoder(videos, masks_enc)

                # Get target representations
                with torch.no_grad():
                    h_target = target_encoder(videos)
                    h_target = F.layer_norm(h_target, h_target.shape[-1:])
                    h_target = apply_masks(h_target, masks_pred)
                    h_target = repeat_interleave_batch(h_target, len(masks_pred))

                # Predict masked patches
                h_pred = predictor(h, masks_enc, masks_pred)

                # Compute loss (L2)
                loss = F.mse_loss(h_pred, h_target)

                if loss_exp != 1.0:
                    loss = loss ** loss_exp

                loss = loss / gradient_accumulation

            # Backward
            if mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (itr + 1) % gradient_accumulation == 0:
                if mixed_precision:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()),
                    clip_grad
                )
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # Update scheduler
                scheduler.step()
                wd_scheduler.step()

                # Update EMA
                with torch.no_grad():
                    m = next(ema)
                    for p_ema, p in zip(target_encoder.parameters(), encoder.parameters()):
                        p_ema.data.mul_(m).add_(p.data, alpha=1 - m)

            # Logging
            loss_meter.update(loss.item() * gradient_accumulation)
            time_meter.update(time.time() - iter_start)

            if (itr + 1) % log_freq == 0 and rank == 0:
                logger.info(
                    f'  [{itr+1}/{ipe}] loss: {loss_meter.avg:.4f}, '
                    f'lr: {optimizer.param_groups[0]["lr"]:.2e}, '
                    f'time: {time_meter.avg:.2f}s'
                )

            # Log attention entropy periodically
            if (itr + 1) % log_attention_every == 0 and rank == 0:
                # This would require extracting attention weights from encoder
                # For now, just log a placeholder
                attention_entropy = -1.0  # Placeholder
                if csv_logger:
                    csv_logger.log({
                        'epoch': epoch + 1,
                        'itr': itr + 1,
                        'loss': loss_meter.avg,
                        'lr': optimizer.param_groups[0]['lr'],
                        'attention_entropy': attention_entropy,
                    })

        # End of epoch logging
        if rank == 0:
            logger.info(f'Epoch {epoch+1} complete. Loss: {loss_meter.avg:.4f}')

            # Save checkpoint
            checkpoint = {
                'encoder': encoder.module.state_dict() if hasattr(encoder, 'module') else encoder.state_dict(),
                'predictor': predictor.module.state_dict() if hasattr(predictor, 'module') else predictor.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None,
                'epoch': epoch + 1,
                'loss': loss_meter.avg,
            }
            ckpt_path = log_folder / f'checkpoint_epoch{epoch+1:03d}.pt'
            torch.save(checkpoint, ckpt_path)
            logger.info(f'Saved checkpoint: {ckpt_path}')

            # Save latest
            torch.save(checkpoint, log_folder / 'checkpoint_latest.pt')

    logger.info('Training complete!')


if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True, help='Config file path')
    args = parser.parse_args()

    with open(args.fname, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
