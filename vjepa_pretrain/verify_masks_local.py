"""
Quick verification that both Endoscapes and SAGES masks are accessible.
"""

from pathlib import Path
import numpy as np
from PIL import Image

# Local mask paths
SAGES_MASKS = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\Masks\synthetic_masks\semantic")
ENDOSCAPES_MASKS = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\Masks\synthetic_masks_sam2")

def count_masks(path, pattern="**/*.png"):
    return len(list(path.glob(pattern)))

def analyze_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    unique, counts = np.unique(mask, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))

print("=" * 60)
print("Local Mask Verification")
print("=" * 60)

# SAGES
print(f"\n1. SAGES Masks: {SAGES_MASKS}")
if SAGES_MASKS.exists():
    sages_count = count_masks(SAGES_MASKS, "*.png")
    print(f"   Count: {sages_count:,} masks")

    sample = list(SAGES_MASKS.glob("*.png"))[0]
    print(f"   Sample: {sample.name}")
    print(f"   Classes: {analyze_mask(sample)}")
else:
    print("   NOT FOUND")

# Endoscapes
print(f"\n2. Endoscapes Masks: {ENDOSCAPES_MASKS}")
if ENDOSCAPES_MASKS.exists():
    for split in ['train', 'val', 'test']:
        split_dir = ENDOSCAPES_MASKS / split / 'semantic'
        if split_dir.exists():
            count = count_masks(split_dir, "*.png")
            print(f"   {split}: {count:,} masks")

            if count > 0:
                sample = list(split_dir.glob("*.png"))[0]
                print(f"      Sample: {sample.name}")
                print(f"      Classes: {analyze_mask(sample)}")
else:
    print("   NOT FOUND")

print("\n" + "=" * 60)
print("Path Summary for Config")
print("=" * 60)
print(f"""
# For local testing, update surgical_vitl16.yaml:

data:
  endoscapes:
    frames_dir: C:/Users/sufia/Documents/Uni/Masters/DISSERTATION/endoscapes
    masks_dir: C:/Users/sufia/Documents/Uni/Masters/DISSERTATION/Masks/synthetic_masks_sam2

  sages:
    frames_dir: C:/Users/sufia/Documents/Uni/Masters/DISSERTATION/sages_cvs_challenge_2025_r1/sages_cvs_challenge_2025/frames
    masks_dir: C:/Users/sufia/Documents/Uni/Masters/DISSERTATION/Masks/synthetic_masks/semantic
""")
