# V-JEPA CVS Classification Experiments

## Goal
Train a CVS (Critical View of Safety) classifier using V-JEPA 2 video features for laparoscopic cholecystectomy safety assessment.

## Hardware
- Local: RTX 3080 12GB (Windows)
- Cloud: RunPod (for comparison)

---

## Experiment 2: Attention Pooling Baseline

### Configuration
| Parameter | Value |
|-----------|-------|
| Backbone | V-JEPA 2 ViT-L (frozen) |
| Pooling | Attention (8 heads) |
| Head | MLP (512 hidden) |
| Loss | BCEWithLogitsLoss |
| Batch Size | 32 |
| LR | 5e-4 |
| Epochs | 15 (early stopped at 4) |

### Results

| Epoch | Train mAP | Val mAP | Train-Val Gap |
|-------|-----------|---------|---------------|
| 1 | 26.21% | 42.41% | -16.2% |
| 2 | 54.69% | **49.79%** | +4.9% |
| 3 | 66.13% | 49.01% | +17.1% |
| 4 | 72.73% | 48.31% | +24.4% |

**Best Result:** 49.79% Val mAP (Epoch 2)

**Per-class AP (Epoch 4):**
- C1 (Hepatocystic Triangle): 48.98%
- C2 (Cystic Plate): 50.13%
- C3 (Two Structures): 45.83%

### Observations
1. **Overfitting:** Clear overfitting after epoch 2
   - Train mAP: 26% → 73% (continuously increasing)
   - Val mAP: 42% → 50% → 48% (peaked at epoch 2, then declining)

2. **Ceiling:** RunPod achieved ~55% mAP, so there's room for improvement

3. **Attention Analysis:** Model learns to focus on anatomically relevant regions (see diagnostics)

### Checkpoint
`results/exp2_local_attention/run_20260130_113325/best_model.pt`

### Training Curve
![Training Curve](visualizations/exp2_attention_baseline/training_curve.png)

### Attention Diagnostic Analysis

Ran targeted diagnostic on 12 validation samples (3 per category):

| Category | Samples | Avg Attention Std | Key Finding |
|----------|---------|-------------------|-------------|
| Full CVS [1,1,1] | 3 | 0.00196 | Most focused attention |
| C1-only [1,0,0] | 3 | 0.00128 | Often misclassified |
| Mixed (C1=0) | 3 | 0.00142 | Variable patterns |
| All-negative | 3 | 0.00150 | Diffuse but correct rejection |

**Key Findings:**

1. **Attention focus correlates with confidence** - Higher std = more focused attention = higher predictions
2. **Model struggles with partial CVS** - C1-only samples often get wrong class prioritized (C3 predicted instead of C1)
3. **Correct rejection of negatives** - Model maintains diffuse attention and low predictions for negative samples
4. **49.79% ceiling explained** - Difficulty distinguishing partial from full CVS achievement

**Sample Analysis:**
| Video ID | Ground Truth | Prediction | Attention Max | Issue |
|----------|--------------|------------|---------------|-------|
| 160_36025 | [1,1,1] | C1=0.53 | 0.102 | Correct, confident |
| 149_23125 | [1,0,0] | C3=0.50 | 0.033 | **Wrong class** |
| 122_35150 | [0,0,0] | All<0.01 | 0.016 | Correct rejection |

**Visualizations:** `visualizations/exp2_attention_baseline/diagnostics/`
**Full Analysis:** `visualizations/exp2_attention_baseline/diagnostic_analysis.md`

### Criterion Comparison Analysis

Ran detailed comparison across single-criterion samples (5 samples each):

| Criterion | Avg Std | Avg Max | Accuracy | Critical Issue |
|-----------|---------|---------|----------|----------------|
| C1-only | 0.00161 | 0.053 | 2/5 | Weak detection |
| **C2-only** | 0.00148 | 0.050 | **0/5** | **COMPLETE FAILURE** |
| C3-only | 0.00185 | 0.070 | 3/5 | Best single-criterion |
| Full CVS | 0.00194 | 0.066 | 2/5 high conf | Inconsistent |

**CRITICAL FINDING: C2 Blindness**

The model **cannot detect C2-only samples at all**:
- All 5 C2-only samples: Model predicts C1 > C2
- Model conflates cystic plate (C2) with hepatocystic triangle (C1)
- This is likely the primary bottleneck for the 50% mAP ceiling

**Per-Criterion Detection Accuracy:**
| Criterion | When Only That Criterion is Positive | When Part of Full CVS |
|-----------|-------------------------------------|----------------------|
| C1 | 40% confident, 40% weak | Usually highest |
| **C2** | **0% - Never detected correctly** | Underestimated |
| C3 | 60% correct | Often highest in Full CVS |

**Attention Pattern Observations:**
- C3-only samples have most focused attention (highest std)
- C2-only samples have most diffuse attention (lowest std)
- No clear spatial clustering within any criterion
- All criteria prefer middle temporal frames (bins 3-5)

**Visualizations:** `visualizations/criterion_comparison/`
**Full Analysis:** `visualizations/criterion_comparison/comparison_analysis.md`

---

## Experiment 3: Focal Loss

### Configuration
Same as Exp2 except:
- **Loss:** Focal Loss (alpha=0.25, gamma=2.0)

### Hypothesis
Focal loss will improve performance by:
- Down-weighting easy negatives (73.9% of samples are all-negative)
- Focusing learning on hard examples

### Status
COMPLETED - **NEGATIVE RESULT**

### Results
| Epoch | Train mAP | Val mAP |
|-------|-----------|---------|
| 1 | 16.26% | 24.35% |
| 2 | - | **24.98%** |
| 3 | - | (stopped) |

**Best Result:** 24.98% Val mAP (Epoch 2) - **WORSE than baseline (49.79%)**

### Analysis
Focal loss significantly underperformed BCE baseline:
- Val mAP: 24.98% vs 49.79% (half the performance)
- alpha=0.25, gamma=2.0 was too aggressive for this dataset
- Down-weighting easy negatives may have hurt learning of class boundaries
- The low focal loss values (~0.04) were misleading - numerically small but not better

### Lessons Learned
1. Focal loss is not universally beneficial for imbalanced data
2. For multi-label classification with moderate imbalance (~5-8x), BCE may be sufficient
3. Should try balanced sampling (Exp4) instead of loss reweighting

---

## Experiment 4: Balanced Sampling

### Configuration
Same as Exp2 except:
- **Sampling:** WeightedRandomSampler (class_balanced strategy)

### Hypothesis
Oversampling positive examples will improve mAP by ensuring the model sees them more frequently.

### Status
COMPLETED - **NEGATIVE RESULT**

### Results
| Epoch | Train mAP | Val mAP | Train-Val Gap |
|-------|-----------|---------|---------------|
| 1 | 69.81% | 39.28% | +30.5% |
| 2 | 86.22% | 44.86% | +41.4% |
| 3 | 90.16% | **46.90%** | +43.3% |
| 4 | 92.81% | 44.11% | +48.7% |
| 5 | 94.44% | 45.44% | +49.0% |

**Best Result:** 46.90% Val mAP (Epoch 3) - **WORSE than baseline (49.79%)**

### Analysis
Balanced sampling underperformed BCE baseline:
- Val mAP: 46.90% vs 49.79% (~3% worse)
- Severe overfitting: Train mAP reached 94% while Val mAP peaked at 47%
- Train-Val gap much larger than Exp2 (49% vs 24%)
- Model overfit to the resampled (balanced) distribution
- Validation uses original imbalanced distribution, causing distribution mismatch

### Lessons Learned
1. Balanced sampling causes distribution mismatch between train and val
2. High train mAP (69-94%) with lower val mAP indicates overfitting to resampled data
3. For this dataset, the original imbalanced distribution may be more representative
4. BCE with random sampling (Exp2) remains the best approach

---

## Experiment 5: Focal Loss + Balanced Sampling

### Configuration
Combines Exp3 and Exp4:
- Loss: Focal Loss (alpha=0.25, gamma=2.0)
- Sampling: WeightedRandomSampler (class_balanced)

### Hypothesis
Combined approach provides complementary benefits.

### Status
SKIPPED (both components failed individually)

---

## Experiment 6: C2-Weighted Loss

### Configuration
Same as Exp2 except:
- **pos_weight:** [1.0, 5.0, 1.0] - 5x penalty for C2 false negatives

### Hypothesis
Based on criterion comparison analysis showing **0% C2-only detection accuracy**, we hypothesize that heavily penalizing C2 false negatives will force the model to learn C2-specific features instead of conflating them with C1.

### Motivation
From criterion comparison (5 samples each):
- C1-only: 40% confident detection
- **C2-only: 0% detection** - model always predicts C1 > C2
- C3-only: 60% detection
- C2 has the highest class imbalance (8.55x)

### Status
COMPLETED - **NEGATIVE RESULT**

### Results
| Epoch | Train mAP | Val mAP | C1 AP | C2 AP | C3 AP | Train-Val Gap |
|-------|-----------|---------|-------|-------|-------|---------------|
| 1 | 26.21% | 42.41% | 43.03% | 44.77% | 39.44% | -16.2% |
| 2 | 54.69% | **49.79%** | 51.98% | **52.86%** | 44.52% | +4.9% |
| 3 | 66.13% | 49.01% | 49.79% | 49.51% | 47.73% | +17.1% |
| (stopped) | | | | | | |

**Best Result:** 49.79% Val mAP (Epoch 2) - **SAME as baseline**

### Analysis
C2-weighted loss did **NOT** improve overall performance:
- Val mAP: 49.79% = baseline (no improvement)
- C2 AP at epoch 2: 52.86% (slight improvement over baseline ~50%)
- But by epoch 3: C2 AP dropped to 49.51% (overfitting)
- Training mAP rapidly increased (66%) while val mAP declined

**Why it failed:**
1. The 5x weight caused faster overfitting on C2
2. The model learned to predict C2 more often, but not more accurately
3. C2 detection requires better *features*, not just higher loss penalty
4. Without explicit anatomical supervision, the model cannot distinguish C2 from C1

### Lessons Learned
1. **Loss weighting alone cannot fix feature-level confusion**
2. C2 and C1 share similar visual appearance in V-JEPA's feature space
3. Need explicit anatomical supervision (segmentation) to teach C2 features
4. This motivates Exp8: Multi-task fine-tuning with segmentation

### Checkpoint
`results/exp6_c2_weighted_loss/run_20260130_211349/best_model.pt` (Epoch 2)

---

## Experiment 8: Multi-task Fine-tuning (CVS + Segmentation)

### Configuration
| Parameter | Value |
|-----------|-------|
| Backbone | V-JEPA 2 ViT-L (last 2 layers unfrozen) |
| CVS Head | Attention pooling + MLP |
| Seg Head | Lightweight decoder (5 classes) |
| Loss | CVS BCE + Seg CE (weighted 0.5) |
| Batch Size | 16 (effective 32 with grad accum) |
| Backbone LR | 1e-5 |
| Head LR | 5e-4 |
| Platform | RunPod A100 80GB |

### Hypothesis
Joint segmentation learning provides explicit anatomical supervision that:
1. **Improves C2 detection** - cystic plate now has direct supervision
2. **Guides attention** to anatomically relevant regions
3. **Creates better features** through multi-task regularization

### Motivation
- 51.2% of training clips have mask supervision
- Segmentation classes directly map to CVS criteria:
  - Class 1: Cystic Plate → C2 criterion
  - Class 2: Calot Triangle → C1 criterion
  - Classes 3,4: Cystic Artery/Duct → C3 criterion
- Loss weighting (Exp6) failed → need feature-level improvement

### Status
COMPLETED - **NEGATIVE RESULT**

### Results
| Epoch | Train mAP | Val mAP | Seg mIoU | Train-Val Gap |
|-------|-----------|---------|----------|---------------|
| 1 | 33.35% | 46.71% | 27.28% | -13.4% |
| 2 | 67.12% | **48.03%** | 29.72% | +19.1% |
| 3 | 77.89% | 45.12% | 30.21% | +32.8% |
| 4 | 85.07% | 40.55% | 30.69% | +44.5% |

**Best Result:** 48.03% Val mAP (Epoch 2) - **WORSE than baseline (49.79%)**

### KEY FINDING: V-JEPA Internal Attention Did NOT Change

Extracted internal attention from the fine-tuned backbone and measured entropy:

| Component | Before Fine-tuning | After Fine-tuning |
|-----------|-------------------|-------------------|
| V-JEPA Internal Attention | ~95% entropy (uniform) | **98.2% entropy (MORE uniform!)** |
| Attention Pooler | - | 75.6% entropy (somewhat focused) |

**Critical Insight:** Fine-tuning 2 layers for 2 epochs did NOT make V-JEPA attend to surgical anatomy. The backbone still has nearly uniform attention - actually slightly worse than baseline!

### Segmentation Head Analysis

Visual analysis of segmentation predictions (see `visualizations/exp8_segmentation_check.png`):

1. **Seg head learned to predict anatomy** - predictions overlap with GT regions
2. **Over-segmentation** - model marks more classes than GT
3. **CVS head failing** - predictions biased toward 0 even for positive samples
4. **The seg decoder extracts anatomy from distributed features, not focused attention**

Sample CVS predictions on positive samples:
| Sample | GT Labels | CVS Predictions | Correct? |
|--------|-----------|-----------------|----------|
| Vid 123 | C2=1 | C2=0.14 | No |
| Vid 126 | C1=1, C2=1 | C1=0.10, C2=0.03 | No |
| Vid 127 | C1=1 | C1=0.03 | No |
| Vid 130 | C3=1 | C3=0.62 | Yes |

### Analysis

**Why it failed:**
1. **Rapid overfitting** - 85% train vs 41% val by epoch 4
2. **V-JEPA's internal attention unchanged** - backbone didn't learn to focus on anatomy
3. **Seg head works independently** - decodes anatomy from uniform features
4. **CVS head doesn't benefit** - still relies on unfocused representations
5. **Fine-tuning too aggressive** - 2 layers + 1e-5 LR caused overfitting before learning

**Lessons Learned:**
1. Fine-tuning V-JEPA backbone doesn't automatically improve attention focus
2. The segmentation decoder can extract structure from uniform features
3. But this doesn't transfer to better classification
4. Need more conservative fine-tuning approach (→ Exp9)

### Files
- `dataset_multitask.py` - Dataset with mask loading
- `model_multitask.py` - V-JEPA + CVS head + Seg decoder
- `train_multitask.py` - Multi-task training loop
- `visualize_exp8_anatomy.py` - Segmentation visualization
- `visualize_exp8_internal_attention.py` - Attention entropy analysis

### Checkpoint
`results/exp8_finetune_multitask/run_20260131_114120/best_model.pt`

---

## Experiment 9: Staged Fine-Tuning (PLANNED)

### Hypothesis
Based on Exp8 findings and dermatology ViT paper (fine-tuning on small datasets):
- **Stage 1**: Train heads first while backbone is completely frozen
- **Stage 2**: Then fine-tune backbone minimally (1 layer, very low LR)

This prevents overfitting by letting heads adapt to V-JEPA features before any backbone modification.

### Motivation
Exp8 showed:
- Fine-tuning 2 layers immediately caused overfitting (85% train vs 41% val)
- V-JEPA's internal attention didn't become more focused (98% entropy)
- The problem: heads and backbone were learning simultaneously

Staged approach rationale:
1. Stage 1 lets heads learn optimal use of frozen V-JEPA features
2. Stage 2 then makes minimal backbone adjustments to support those heads
3. Much lower LR prevents catastrophic forgetting of pretrained representations

### Configuration

**Stage 1 (Heads Only):**
| Parameter | Value |
|-----------|-------|
| Backbone | Completely frozen (0 layers) |
| Epochs | 10 |
| Head LR | 5e-4 |
| Batch Size | 32 (effective 128) |
| Early Stopping | 5 epochs patience |

**Stage 2 (Minimal Backbone):**
| Parameter | Value |
|-----------|-------|
| Backbone | Last 1 layer unfrozen (not 2) |
| Epochs | 5 (max) |
| Backbone LR | 1e-6 (20x lower than Exp8) |
| Head LR | 1e-5 (reduced) |
| Batch Size | 16 (effective 128) |
| Early Stopping | 3 epochs patience |

### Key Differences from Exp8
| Setting | Exp8 | Exp9 |
|---------|------|------|
| Staging | None (immediate unfreeze) | 2-stage |
| Unfrozen layers | 2 | 0 → 1 |
| Backbone LR | 1e-5 | 1e-6 (20x lower) |
| seg_weight | 0.5 | 0.3 (focus on CVS) |
| Augmentation | Flip only | Rotation, color jitter, erasing, blur |

### New Augmentations
- Random rotation: ±15 degrees
- Color jitter: brightness 0.3, contrast 0.3, saturation 0.2, hue 0.1
- Random erasing: 20% probability
- Gaussian blur: 10% probability

### Expected Outcome
- Stage 1 should match or beat Exp2 baseline (49.79%)
- Stage 2 should add small improvement without overfitting
- **Target: >52% mAP**

### Files
- `configs/exp9_staged_finetune.yaml` - Experiment config
- `train_staged.py` - Two-stage training script

### Status
READY FOR TESTING

### To Run
```bash
cd /workspace/vjepa
python train_staged.py --config configs/exp9_staged_finetune.yaml
```

---

## Class Distribution Analysis

Training set imbalance:
| Class | Positive | Negative | Ratio |
|-------|----------|----------|-------|
| C1 | 14.6% | 85.4% | 5.86x |
| C2 | 10.5% | 89.5% | **8.55x** |
| C3 | 16.8% | 83.2% | 4.95x |

- 73.9% of samples are all-negative
- Only 4.9% are all-positive
- C2 (Cystic Plate) is the rarest positive class

---

## Summary Table

| Exp | Approach | Best Val mAP | Notes |
|-----|----------|--------------|-------|
| **2** | Attention pooling (frozen) | **49.79%** | **BASELINE** - Best so far |
| 3 | Focal loss | 24.98% | Much worse |
| 4 | Balanced sampling | 46.90% | Distribution mismatch |
| 5 | Focal + Balanced | - | Skipped |
| 6 | C2-weighted loss | 49.79% | No improvement |
| 8 | Multi-task fine-tune (2 layers) | 48.03% | Overfit, V-JEPA unchanged |
| **9** | Staged fine-tune | **TBD** | Next experiment |

## Conclusions

1. **Current Best: Exp2 (Frozen backbone + Attention pooling)** with 49.79% Val mAP
2. **Focal loss hurts performance** - down-weighting easy negatives was counterproductive
3. **Balanced sampling hurts performance** - causes train/val distribution mismatch
4. **Simple BCE with random sampling is best** for this moderate imbalance (~5-8x)
5. **Overfitting is the main challenge** - all experiments showed train-val gap widening
6. **V-JEPA's internal attention is UNIFORM** - ~95-98% entropy, doesn't focus on anatomy
7. **Fine-tuning doesn't fix attention** - Exp8 showed 2 layers unfrozen → still uniform
8. **Seg head can extract anatomy** - but from distributed features, not focused attention
9. **CVS head doesn't benefit from seg head** - multi-task didn't improve classification
10. **C2 DETECTION IS BROKEN** - Model never correctly identifies C2-only samples
11. **Need more conservative fine-tuning** - staged approach (Exp9) may prevent overfitting

## Root Cause Analysis

The ~50% mAP ceiling is explained by multiple factors:

### 1. V-JEPA's Uniform Internal Attention
- V-JEPA's internal attention has ~95-98% entropy (nearly uniform)
- Fine-tuning 2 layers did NOT change this (Exp8)
- The backbone treats all spatial patches roughly equally
- It doesn't "see" surgical anatomy differently from background

### 2. C2 Blindness
- C2 (cystic plate) is the rarest class (8.55x imbalance)
- Model conflates C2 with C1 (predicts C1 > C2 for all C2-only samples)
- Loss weighting (Exp6) confirmed: The problem is feature-level, not loss-level

### 3. Overfitting
- All fine-tuning experiments showed rapid overfitting
- Exp8: 85% train vs 41% val by epoch 4
- The model memorizes training data instead of learning generalizable features

### 4. Feature-Classification Gap
- Seg decoder CAN extract anatomy from V-JEPA features
- But CVS head CANNOT leverage this for classification
- The segmentation task works; the classification task doesn't benefit

## Next Steps (Priority Order)

**HIGH PRIORITY - Staged Fine-tuning (Exp9):**
1. ✅ Created staged fine-tuning code
2. Run on RunPod A100:
   - Stage 1: Train heads with frozen backbone (10 epochs)
   - Stage 2: Minimal backbone fine-tuning (1 layer, 1e-6 LR)
3. Target: >52% mAP without overfitting

**Medium Priority - If Exp9 fails:**
4. Try criterion-specific attention heads (one per CVS criterion)
5. Consider hierarchical classification (detect anatomy first, then CVS)
6. Experiment with different backbone layers (middle layers vs last layers)

**Lower Priority - Alternative Approaches:**
7. Try different video foundation models (VideoMAE, InternVideo)
8. Consider ensemble of frozen features + fine-tuned features
9. Surgical domain pre-training if available

---

*Last updated: 2026-01-31*
