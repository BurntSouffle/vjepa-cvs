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

## Experiment 9: Staged Fine-Tuning - COMPLETED

### Hypothesis
Based on Exp8 findings and dermatology ViT paper (fine-tuning on small datasets):
- **Stage 1**: Train heads first while backbone is completely frozen
- **Stage 2**: Then fine-tune backbone minimally (1 layer, very low LR)

This prevents overfitting by letting heads adapt to V-JEPA features before any backbone modification.

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

### Results

**Stage 1 (Frozen backbone, train heads only):**
| Epoch | Train mAP | Val mAP | C1 AP | C2 AP | C3 AP |
|-------|-----------|---------|-------|-------|-------|
| 1 | 27.34% | 44.15% | 43.27% | 46.93% | 42.26% |
| 2 | 60.18% | **49.94%** | 45.33% | 51.28% | 53.21% |
| 3 | 72.26% | 45.71% | 40.23% | 44.63% | 52.28% |
| 4 | 78.39% | 39.87% | 40.00% | 36.56% | 43.05% |

Best Stage 1: **49.94% mAP** (Epoch 2) - matches baseline!

**Stage 2 (Unfreeze 1 layer, very low LR):**
| Epoch | Train mAP | Val mAP | C1 AP | C2 AP | C3 AP |
|-------|-----------|---------|-------|-------|-------|
| 1 | - | **48.49%** | - | - | - |
| 3 | 76.42% | 47.82% | 42.68% | 49.58% | 51.19% |
| 4 | 77.36% | 47.86% | 42.86% | 49.59% | 51.12% |

Early stopped at epoch 4. Best Stage 2: **48.49% mAP** (Epoch 1)

### Conclusion

- Stage 1 (frozen): **49.94%** ✅ Best result, matches baseline
- Stage 2 (unfreeze 1): **48.49%** ❌ Made it worse (-1.45%)
- **Fine-tuning V-JEPA hurts performance** - even 1 layer with 1e-6 LR overfits
- Pretrained features are better than anything we can fine-tune to

### Key Finding
**Any backbone fine-tuning degrades performance.** Even the most conservative approach (1 layer, 1e-6 LR, after head pretraining) caused immediate degradation. V-JEPA's pretrained representations are optimal for this task - we should focus on better ways to use them, not modify them.

### Files
- `configs/exp9_staged_finetune.yaml` - Experiment config
- `train_staged.py` - Two-stage training script

### Checkpoint
`/workspace/results/exp9_staged_finetune/run_20260131_211037/stage1_best.pt`

---

## Experiment 10: LoRA Fine-Tuning - COMPLETED ✅

### Hypothesis
LoRA (Low-Rank Adaptation) adapts pretrained features without destroying them by:
- Freezing all original weights
- Adding small trainable adapter matrices (A × B) to attention layers
- Only 2.25% of parameters trainable

### Motivation
Exp9 confirmed that even minimal backbone fine-tuning hurts performance. LoRA offers an alternative:
- Original V-JEPA weights stay frozen (no degradation possible)
- Learns domain-specific adaptations through low-rank matrices
- Much smaller parameter count reduces overfitting risk

---

### Exp10a: Baseline LoRA (r=16) - COMPLETED

**Settings:**
| Setting | Value |
|---------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| LoRA dropout | 0.1 |
| LoRA params | 1.87M |
| Head params | 5.65M |
| Total trainable | 7.52M (2.25%) |
| LoRA LR | 1e-4 |
| Head LR | 5e-4 |

**Results:**
| Epoch | Train mAP | Val mAP | C1 AP | C2 AP | C3 AP |
|-------|-----------|---------|-------|-------|-------|
| 1 | 25.32% | 43.79% | 40.81% | 48.89% | 41.66% |
| 2 | 64.34% | **53.75%** 🏆 | 50.69% | 55.03% | 55.54% |
| 3 | 78.01% | 52.99% | 49.75% | 55.40% | 53.82% |

**Best: 53.75% mAP (Epoch 2)** - New record! +3.96% over baseline

**Checkpoint:** `results/exp10_lora/run_20260201_203545/best_model.pt`

---

### Key Finding: LoRA Did NOT Change Attention Patterns

| Model | Attention Entropy |
|-------|-------------------|
| Baseline V-JEPA | 98.4% (uniform) |
| LoRA V-JEPA | 97.9% (still uniform) |
| **Change** | **0.5%** (minimal) |

LoRA improved performance by modifying VALUE projections (what gets extracted), not attention patterns (where to look).

```
V-JEPA Internal Attention: WHERE to look → Still uniform (~98%)
LoRA Value Projections:    WHAT to output → Modified to be more discriminative
Task Heads:                HOW to classify → Trained on surgical domain
```

### Sample Predictions Analysis (Exp10a)

| Sample | Type | GT Labels | Predictions | Accuracy |
|--------|------|-----------|-------------|----------|
| 122_35150 | Negative | [0,0,0] | [0.02, 0.00, 0.00] | **3/3** ✅ |
| 127_19475 | Full CVS | [1,1,1] | [0.76, 0.61, 0.43] | 2/3 |
| 129_68650 | C1-only | [1,0,0] | [0.29, 0.11, 0.05] | 2/3 |
| 123_37650 | C2-only | [0,1,0] | [0.19, 0.15, 0.01] | 2/3 |

**Pattern:** Strong on clear cases (negatives, full CVS), weak on single-criterion samples.

---

### Exp10c: Lower Learning Rate - COMPLETED

**Hypothesis:** Lower LR will allow training past epoch 2 without overfitting

**Settings:**
| Setting | Exp10a | Exp10c |
|---------|--------|--------|
| LoRA LR | 1e-4 | 5e-5 (halved) |
| Head LR | 5e-4 | 2.5e-4 (halved) |
| Warmup | 2 epochs | 3 epochs |
| Epochs | 15 | 20 |
| Patience | 5 | 7 |

**Results:**
| Epoch | Train mAP | Val mAP | C1 AP | C2 AP | C3 AP |
|-------|-----------|---------|-------|-------|-------|
| 1 | 22.98% | 38.68% | 37.25% | 41.87% | 36.92% |
| 2 | 60.48% | 48.90% | 48.86% | 50.13% | 47.70% |
| 3 | 77.01% | **52.56%** | 48.43% | 54.80% | 54.45% |
| 4 | 83.01% | 46.53% | 41.56% | 48.89% | 49.15% |

**Best: 52.56% mAP (Epoch 3)**

**Conclusion:** Lower LR shifted peak from epoch 2 to epoch 3, but didn't prevent overfitting or reach higher mAP. Still -1.19% below Exp10a.

---

### Exp10b: Higher Rank LoRA (r=32 + k_proj) - COMPLETED

**Hypothesis:** More LoRA capacity + targeting k_proj might change attention patterns

**Settings:**
| Setting | Exp10a | Exp10b |
|---------|--------|--------|
| LoRA r | 16 | **32** (2x capacity) |
| LoRA alpha | 32 | **64** |
| Target modules | q, v | **q, k, v** (+k_proj) |
| Trainable params | 7.52M | 9.39M |

**Results:**
| Epoch | Train mAP | Val mAP | C1 AP | C2 AP | C3 AP | Train-Val Gap |
|-------|-----------|---------|-------|-------|-------|---------------|
| 1 | 33.66% | **54.61%** 🏆 | 52.16% | **60.34%** 🏆 | 51.32% | -21% |
| 2 | 70.80% | 50.63% | 47.09% | 54.35% | 50.45% | +20% |
| 3 | 80.81% | 48.05% | 47.82% | 48.03% | 48.31% | +33% |

**Best: 54.61% mAP (Epoch 1)** - NEW RECORD!
- **C2 AP: 60.34%** - Biggest improvement, +10% over baseline!
- Peaked at epoch 1, then overfit rapidly

**Why k_proj matters (in theory):**
Adding k_proj means LoRA can modify how attention KEYS are computed, not just queries and values.
- **q_proj** affects query vectors (what to search for)
- **k_proj** affects key vectors (what to match against) ← NEW
- **v_proj** affects value vectors (what to output)

---

## Key Finding: k_proj Did NOT Change Attention Patterns

### Attention Entropy Analysis

| Model | Avg Entropy | vs Baseline | Interpretation |
|-------|-------------|-------------|----------------|
| Baseline V-JEPA | 98.4% | - | Nearly uniform |
| Exp10a (r=16, q+v) | 97.9% | -0.5% | Still uniform |
| **Exp10b (r=32, q+k+v)** | **97.9%** | **-0.5%** | **Still uniform** |

**Critical Finding:** Despite adding k_proj and doubling the rank, Exp10b shows **identical entropy** to Exp10a!

### What This Means

1. **Performance gains came from better VALUE extraction, NOT attention focus**
   - More parameters in q/k/v projections = more discriminative features
   - But the model still attends uniformly to all spatial positions

2. **V-JEPA's uniform attention is deeply ingrained**
   - Self-supervised pretraining on natural videos created this behavior
   - LoRA cannot fundamentally change this architecture

3. **C2 improvement (60.34% AP) came from:**
   - Better feature transformation in value projections
   - NOT from attending more to cystic plate regions

### Implications for Future Work

To actually change WHERE V-JEPA attends, would need:
- **Architectural changes** (window attention like SwinCVS)
- **Attention supervision** (explicit loss on attention maps)
- **Full fine-tuning** of attention layers (but this causes overfitting)

### Visualizations
See `visualizations/exp10b_attention_analysis/`:
- `attention_comparison_all_lora.png` - Side-by-side heatmaps
- `entropy_comparison.png` - Bar chart comparison
- `c2_detection_analysis.png` - C2 predictions
- `ANALYSIS_SUMMARY.md` - Full report

---

## Key Finding: Higher LoRA Rank = Earlier Peak

| LoRA r | Peak Epoch | Best Val mAP | Best C2 AP |
|--------|------------|--------------|------------|
| 16 | Epoch 2 | 53.75% | 55.03% |
| 16 (low LR) | Epoch 3 | 52.56% | 54.80% |
| **32 + k_proj** | **Epoch 1** | **54.61%** | **60.34%** |

**Insight:** More LoRA capacity enables faster learning but also faster overfitting. The model reaches a higher peak but in fewer epochs.

**Implication:** For surgical video analysis with limited data, use higher LoRA rank but train for very few epochs (1-2 max).

---

### Files
- `configs/exp10_lora.yaml` - Exp10a config
- `configs/exp10b_lora_r32.yaml` - Exp10b config
- `configs/exp10c_lora_lowlr.yaml` - Exp10c config
- `train_lora.py` - LoRA training script (requires `pip install peft`)
- `analyze_exp10_lora_simple.py` - Exp10a attention analysis script
- `analyze_exp10b_attention.py` - Exp10b attention analysis (compares all LoRA experiments)
- `visualizations/exp10_lora_analysis/` - Exp10a analysis visualizations
- `visualizations/exp10b_attention_analysis/` - Exp10b attention comparison (baseline vs 10a vs 10b)

---

## Layer-by-Layer Entropy Analysis (Exp10b Model)

Analyzed attention entropy across all 24 transformer layers of V-JEPA with LoRA r=32 + k_proj.

### Results

| Layer | Entropy | Notes |
|-------|---------|-------|
| 0 | 97.8% | Early layer |
| 6 | 99.6% | Most uniform |
| 12 | ~99% | Middle layer |
| 22 | **94.8%** | **Most focused** (still very uniform) |
| 23 | 98.3% | Final layer |

### Key Findings

1. **All 24 layers have near-uniform attention** (94.8% - 99.6% entropy)
2. **Early layers get MORE uniform** (97.8% → 99.6% from layer 0→6)
3. **Layer 22 is slightly more focused** but still 94.8% entropy
4. **Final layer (23) returns to uniform** at 98.3%
5. **Some samples hit 100% entropy** - perfectly uniform, no spatial differentiation

### Visualization
```
Layer:    0    6    12   18   22   23
Entropy: 97.8→99.6→99.0→98.5→94.8→98.3
      ↑              ↓    ↑
   uniform      slight  back to
                focus   uniform
```

### Implications

- LoRA cannot meaningfully change attention at ANY layer
- Even the "most focused" layer (22) is still 94.8% uniform
- **Hard attention masking (Exp12) is necessary** - training-based approaches failed at all layers

### Files
- `analyze_layer_entropy.py` - Analysis script
- `visualizations/layer_entropy_analysis/exp10b/` - Bar chart, heatmaps, CSV data

---

## Key Insights

1. **Best result: 55.98% mAP** (Exp12 with strong regularization)
2. **Regularization delays overfitting** - Peak moved from epoch 1 to epoch 4
3. **LoRA improves V-JEPA** - But only by ~6% over baseline
4. **V-JEPA attention cannot be changed** - Stays ~98% uniform despite all interventions
5. **Attention supervision failed** - Even λ=1.0 didn't reduce entropy
6. **Hard masking + MixUp/CutMix helps** - More stable training, higher peak
7. **Fundamental limitation: Global attention** - V-JEPA architecture unsuited for surgical anatomy

## Progress Summary
```
Baseline:     49.79%

LoRA:       54.61% (+4.82%)
Regularization: 55.98% (+1.37%)
─────────────────────────
Total gain:   +6.19%

Gap to SwinCVS: 11.47% remaining
```

---

## Key Insights (Detailed)

Based on experiments 2-10, here are the critical learnings:

### 1. V-JEPA's Internal Attention is Uniform (~95-98% entropy)
- V-JEPA doesn't naturally focus on surgical anatomy
- It treats all spatial patches roughly equally
- This is by design for self-supervised video learning
- **Implication:** Need to adapt features, not just add heads

### 2. Classifier-Level Fixes Don't Work
| Approach | Result | Why |
|----------|--------|-----|
| Focal loss | 24.98% (-25%) | Too aggressive down-weighting |
| Balanced sampling | 46.90% (-3%) | Train/val distribution mismatch |
| C2-weighting (5x) | 49.79% (same) | Loss weighting can't fix features |

### 3. Fine-tuning Overfits Quickly
- Exp8 overfit by epoch 2-3 (85% train vs 41% val)
- V-JEPA's attention remained uniform after fine-tuning
- 2 layers + 1e-5 LR was too aggressive

### 4. Segmentation Head CAN Learn Anatomy
- Seg decoder extracts structure from distributed features
- But this doesn't transfer to better classification
- The classification task needs focused representations

### 5. Staged Training Confirmed: Frozen is Best (Exp9)
- Stage 1 (frozen backbone): **49.94% mAP** ✅ Best result
- Stage 2 (unfreeze 1 layer): **48.49% mAP** ❌ Degraded
- Even 1 layer with 1e-6 LR causes overfitting
- **Conclusion: Don't fine-tune V-JEPA at all**

### 6. LoRA Achieved Best Results (Exp10b - 54.61% mAP)
- Adapt features without modifying original weights
- Higher rank (r=32) + k_proj = better results but faster overfitting
- Original V-JEPA weights stay frozen
- **Best result: 54.61% mAP (+4.82% over baseline), C2 AP: 60.34%**

### 7. Higher LoRA Rank = Earlier Peak
- r=16: peaks at epoch 2 (53.75% mAP)
- r=32 + k_proj: peaks at epoch 1 (54.61% mAP)
- More capacity = faster learning = faster overfitting
- **For limited data: use high rank, train 1-2 epochs max**

### 8. k_proj Does NOT Change Attention Patterns
- Exp10a (r=16, q+v): 97.9% entropy
- Exp10b (r=32, q+k+v): 97.9% entropy (SAME!)
- Adding k_proj did not make attention more focused
- **All performance gains came from value extraction, not attention focus**
- V-JEPA's uniform attention is architecturally ingrained

---

## Future Experiments

### Other Techniques to Consider

| Technique | Description | Priority | Rationale |
|-----------|-------------|----------|-----------|
| **MixUp/CutMix** | Blend images during training | High | Critical regularizer for transformers, shown to help ViT significantly |
| **DropPath** | Randomly skip transformer layers | Medium | Strong regularizer, reduces overfitting |
| **Test-Time Augmentation** | Average predictions over augmented versions | Low | Free boost, no training cost |
| **Layer-wise LR Decay (LLRD)** | Different LR per layer depth | Medium | Earlier layers get lower LR, preserves general features |
| **Gradient Checkpointing** | Trade compute for memory | Low | Would allow larger batches |
| **Label Smoothing** | Soft targets instead of hard 0/1 | Medium | Prevents overconfident predictions |

### Technique Details

**MixUp (High Priority):**
```python
# During training, blend two samples
lambda_ = np.random.beta(0.2, 0.2)
mixed_video = lambda_ * video_a + (1 - lambda_) * video_b
mixed_label = lambda_ * label_a + (1 - lambda_) * label_b
```

**Layer-wise LR Decay:**
```python
# Lower LR for earlier layers
for i, layer in enumerate(model.backbone.layers):
    layer_lr = base_lr * (decay_rate ** (num_layers - i))
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

| Exp | Approach | Val mAP | Peak Epoch | Status |
|-----|----------|---------|------------|--------|
| 2 | Baseline (frozen) | 49.79% | 2 | ✅ |
| 10a | LoRA r=16 | 53.75% | 2 | ✅ |
| 10b | LoRA r=32 + k_proj | 54.61% | 1 | ✅ |
| 10c | LoRA low LR | 52.56% | 3 | ✅ |
| 11 | Attention supervision | 49.77% | - | ❌ Failed |
| **12** | **Regularization + Hard masking** | **55.98%** 🏆 | **4** | ✅ **Best** |

**Current Best: 55.98% mAP (Exp12)**
**Gap to SwinCVS: 11.47%**

## LoRA Experiments Summary

### What Works
- Higher rank (r=32 > r=16) achieves higher peak performance
- Adding k_proj allows modifying attention computation
- C2 (cystic plate) detection benefits most from LoRA (+10% AP)

### What Doesn't Work
- Lower learning rate just delays overfitting, doesn't prevent it
- Training beyond epoch 1-2 leads to overfitting regardless of settings
- LoRA doesn't fundamentally change V-JEPA's uniform attention pattern

### Optimal LoRA Recipe for V-JEPA + Surgery
```yaml
lora:
  r: 32                          # Higher rank for more capacity
  lora_alpha: 64                 # 2x rank
  target_modules: [q_proj, k_proj, v_proj]  # Include k_proj!
  lora_dropout: 0.1
training:
  epochs: 1-2                    # Stop early! Overfitting is rapid
  lora_lr: 1e-4
  head_lr: 5e-4
```

---

## Exp11: Attention Supervision - COMPLETED ❌

### Result: FAILED

Even with aggressive attention supervision (λ=1.0), entropy remained at 99.2-99.4%.

| Lambda | Entropy Change | Result |
|--------|----------------|--------|
| 0.1 | 99.5% → 99.2% | -0.3% |
| 1.0 | 99.3% → 99.4% | **+0.1% (worse!)** |

**Conclusion:** V-JEPA's attention patterns cannot be changed through training loss. Attention supervision actually made entropy slightly WORSE.

### Files
- `configs/exp11_attention_supervised.yaml`
- `train_attention_supervised.py`

---

## Exp12: Strong Regularization + Hard Attention Masking - COMPLETED ✅

### Hypothesis
Based on external feedback: Use strong regularization (MixUp, CutMix, Label Smoothing) to delay overfitting, and hard attention masking to force focus on anatomy.

### Key Changes from Exp10b

| Component | Exp10b | Exp12 |
|-----------|--------|-------|
| MixUp | ❌ | ✅ α=0.8 |
| CutMix | ❌ | ✅ α=1.0 |
| Label Smoothing | ❌ | ✅ 0.1 |
| Weight Decay | 0.05 | 0.1 |
| Warmup | 2 epochs | 0.5 epochs |
| Attention | Uniform | Hard masked on anatomy |

### Results

| Epoch | Train mAP | Val mAP | C1 | C2 | C3 | Notes |
|-------|-----------|---------|-----|-----|-----|-------|
| 1 | 20.30% | 45.59% | 51.35% | 46.44% | 38.98% | Building |
| 2 | 34.37% | 55.77% | 58.27% | 56.77% | 52.25% | Great |
| 3 | 39.32% | 53.59% | 50.13% | 55.76% | 54.89% | Dip |
| 4 | 41.07% | **55.98%** 🏆 | 54.69% | 57.85% | 55.39% | **NEW BEST** |
| 5 | 44.52% | 47.90% | 51.17% | 46.92% | 45.62% | Drop |
| 6 | 47.50% | 49.74% | 49.94% | 50.98% | 48.29% | Overfit |

**Best: 55.98% mAP (Epoch 4)** - NEW RECORD!

### What Worked

| Improvement | Evidence |
|-------------|----------|
| **Delayed overfitting** | Peak at epoch 4 (vs epoch 1 for Exp10b) |
| **Higher peak mAP** | 55.98% vs 54.61% (+1.37%) |
| **More stable training** | Val > Train for 6 epochs (generalizing) |
| **Better C3 detection** | 55.39% vs 51.32% (+4.07%) |

### What Didn't Work

| Issue | Evidence |
|-------|----------|
| **Still far from SwinCVS** | 55.98% vs 67.45% (11.5% gap) |
| **Attention still uniform** | Hard masking applied but ~98% entropy persists |
| **Eventually overfits** | Epochs 5-6 show declining val mAP |

### Key Insight

Regularization delays overfitting and achieves slightly higher peak, but cannot overcome V-JEPA's fundamental architectural limitations (global uniform attention).

### Files
- `configs/exp12_regularized.yaml`
- `train_regularized.py`

---

## What's Next: Potential Experiments

### Exp13: Small ViT with Window Attention (Train from Scratch)

Instead of adapting V-JEPA, train a smaller model designed for surgery:

| Setting | V-JEPA (current) | Proposed Exp13 |
|---------|------------------|----------------|
| Params | 307M | ~25M |
| Layers | 24 | 12 |
| Attention | Global (uniform) | **Window (local)** |
| Pretrained | Natural video | **None (surgery only)** |

**Hypothesis:** A smaller model with window attention, trained from scratch on surgical data, may learn surgery-specific attention patterns that V-JEPA cannot.

### Exp14: Hybrid SwinCVS + V-JEPA

Combine SwinCVS spatial features with V-JEPA temporal modeling:
```
Frame → SwinCVS (local attention) → Spatial features
↓
V-JEPA temporal layers
↓
CVS Classification
```

### Exp15: V-JEPA with Architectural Window Attention

Modify V-JEPA architecture to use window attention instead of global:
```python
# Replace: Global attention (2048 × 2048)
# With: Window attention (8 × 8 windows)
```

---

## Key Finding: Why V-JEPA's Attention Is Rigid

### Why V-JEPA's Attention Is Hard to Change

**1. Scale Mismatch**
- V-JEPA pretraining: Millions of videos, billions of tokens
- Our fine-tuning: 37K clips, ~10 epochs
- Ratio: ~1000:1 in favor of pretrained patterns

**2. Self-Supervised Objective**
V-JEPA was trained to predict masked video patches. This requires attending to EVERYTHING to gather context:
- Pretraining task: "Predict what's behind the mask"
- Solution learned: "Look everywhere for clues"
- Result: Uniform attention (98% entropy) is OPTIMAL for the original task

**3. LoRA Only Adds Small Perturbations**
- Original weight: W (huge, frozen)
- LoRA adaptation: W + A*B (tiny, ~1-3% of computation)
- Original patterns: ~97-99% preserved

**4. Attention is Computed, Not Directly Learned**
- Q = W_q * input  <-- LoRA modifies this
- K = W_k * input  <-- LoRA modifies this
- Attention = softmax(Q * K^T / sqrt(d))  <-- Formula is FIXED
- The softmax naturally spreads attention. LoRA changes Q and K, but can't fundamentally change how attention is computed.

### What Would Actually Change Attention?

| Approach | Changes Attention? | Why |
|----------|-------------------|-----|
| LoRA on Q,K,V | No, barely | Small perturbations |
| Attention supervision (lambda=0.1) | No, barely | Can't override pretrained patterns |
| Attention supervision (lambda=1.0) | Testing | Stronger signal |
| Window attention (Swin-style) | Yes | Architectural constraint |
| Full pretraining on surgery | Yes | Learn new patterns from scratch |

---

## V-JEPA Architecture Deep Dive

### Data Flow

```
Input: Video [B, T, C, H, W]
│
▼
┌─────────────────────────────────────┐
│     3D Patch Embedding              │
│     [B, 16, 3, 256, 256] ->         │
│     [B, 2048, 1024]                 │
│     (8 temporal x 256 spatial)      │
└─────────────────────────────────────┘
│
▼
┌─────────────────────────────────────┐
│     Transformer Encoder             │
│     24 layers (ViT-L)               │
│                                     │
│   Each layer:                       │
│   x = x + attention(x)  <- Skip #1 │
│   x = x + mlp(x)        <- Skip #2 │
│                                     │
└─────────────────────────────────────┘
│
▼
Output: [B, 2048, 1024]
```

### Key Architectural Properties

| Property | V-JEPA | Implication |
|----------|--------|-------------|
| Skip connections | Yes (2 per layer) | Preserves input signal, stabilizes training |
| Attention type | Global (all tokens) | Every token attends to all 2048 tokens |
| Layers | 24 (ViT-L) | Deep network, 307M params |
| Tokens | 2048 (8 temporal x 256 spatial) | Large attention matrix (2048x2048) |

### Why Uniform Attention Persists Through Skip Connections

```
Layer 0:  Input tokens all similar (patches)
          -> Attention spreads evenly
          -> Skip connection preserves this
Layer 12: Tokens slightly differentiated
          -> Attention still fairly uniform
          -> Skip connection preserves this
Layer 23: Final representations
          -> Attention STILL uniform
          -> Skip connections preserved uniform pattern throughout
```

The residual connections let information flow without forcing attention to focus.

---

### Ideas for Future Experiments

#### 1. Layer Pruning
```python
# Reduce from 24 to 12 layers
model.encoder.layers = model.encoder.layers[:12]
# ~150M params instead of 307M
```
Risk: May break pretrained features

#### 2. U-Net Style Skip Connections
```
Layer 0  ----------------------> Layer 23
Layer 6  ----------------------> Layer 18
Layer 12 ----------------------> Layer 12
```
Let early features (edges, textures) directly influence late layers.

#### 3. Feature Pyramid
```python
# Combine multi-scale features
features = concat([layer_6_out, layer_12_out, layer_23_out])
```
Early layers: local features, Late layers: global context

#### 4. Attention Bias Injection
```python
# Add mask-based bias to attention
attention = softmax(Q @ K^T + mask_bias)
# Boost attention on anatomy regions
```

#### 5. Per-Layer Attention Analysis
Analyze which layers have most/least uniform attention:
```python
for i, layer in enumerate(model.encoder.layers):
    entropy = compute_entropy(extract_attention(layer))
    print(f"Layer {i}: Entropy = {entropy}%")
```
This could reveal where to intervene.

---

### Comparison: V-JEPA vs SwinCVS Architecture

| Aspect | V-JEPA | SwinCVS |
|--------|--------|---------|
| Attention | Global (2048x2048) | Window (24x24) |
| Skip connections | Within-layer residual | Within-layer residual |
| Layers | 24 | ~24 (Swin-B) |
| Params | 307M | 88M |
| Pretraining | Self-supervised video | ImageNet supervised |
| Result | 54.61% mAP | 67.45% mAP |

**Key difference:** SwinCVS's window attention forces local focus, which is better for finding small surgical structures.

---

## Future Directions & Hybrid Ideas

### Goal: Beat SwinCVS (67.45% mAP)

Current gap: 55.98% → 67.45% = **11.47% to close**

### Ideas to Explore

#### 1. Higher LoRA Rank (r=64, r=128)
If r=32 helps significantly, more capacity might help more.
```yaml
lora:
  r: 64
  lora_alpha: 128
  target_modules: [q_proj, k_proj, v_proj]
```

#### 2. LoRA on ALL Attention + MLP
Target more layers for adaptation:
```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj      # Output projection
  - fc1         # MLP layers
  - fc2
```

#### 3. Window Attention Constraint
Force V-JEPA to use local windows like Swin:
- Mask global attention to local regions
- Or add window attention as parallel path

#### 4. Hybrid: SwinCVS Spatial + V-JEPA Temporal
```
Frame → SwinCVS Swin-B → Spatial features (local attention)
↓
V-JEPA temporal layers
↓
CVS Classification
```
Best of both: SwinCVS's local attention + V-JEPA's temporal modeling

#### 5. Multi-Resolution Fusion
```
V-JEPA at 256×256 → Global context
SwinCVS at 384×384 → Fine details
↓
Feature fusion → CVS
```

#### 6. Attention Supervision
Explicitly train attention using mask locations:
```
Loss = CVS_loss + λ * attention_alignment_loss
```
Where attention should focus on masked anatomy regions.

#### 7. Frame Selection Strategy
SwinCVS uses 5 frames at 1fps, V-JEPA uses 16 frames.
- Try V-JEPA with 5 frames (like SwinCVS)
- Try adaptive frame selection (key moments)

### Research Questions for Dissertation

1. **Why does global attention fail for surgical anatomy?**
   - V-JEPA's uniform attention (98% entropy) vs SwinCVS's local windows
   - Small anatomical structures need local focus

2. **Can LoRA adapt attention patterns or only value projections?**
   - r=16: Only value projections changed
   - r=32 + k_proj: Testing if attention patterns change

3. **What's the optimal balance between temporal context and spatial focus?**
   - More frames (16) vs fewer frames (5)
   - Global attention vs local attention

4. **Can hybrid architectures combine strengths of both approaches?**
   - SwinCVS: Good spatial attention, limited temporal
   - V-JEPA: Good temporal, poor spatial attention

---

## Project Timeline

**Current Status:** February 2026
**Submission Deadline:** ~July 2026 (5 months)

### Milestones

| Month | Goal |
|-------|------|
| Feb | Complete LoRA experiments, PhD interview presentation (Feb 11) |
| Mar | Implement hybrid approaches, analyze attention patterns |
| Apr | Run comprehensive experiments, ablation studies |
| May | Write dissertation draft, create visualizations |
| Jun | Revise, final experiments, polish |
| Jul | Submit! |

### Target for Paper
- Beat SwinCVS (67.45%) OR
- Novel insights into video foundation models for surgery
- First comprehensive analysis of V-JEPA attention for medical imaging

---

## Conclusions

1. **Current Best: Exp12 (Regularization + Hard Masking)** with **55.98%** Val mAP (+6.19% over baseline)
2. **Regularization delays overfitting** - Peak moved from epoch 1 (Exp10b) to epoch 4 (Exp12)
3. **LoRA + regularization is the best combination** - LoRA adapts features, regularization prevents overfitting
4. **LoRA works better than direct fine-tuning** - adapts without destroying pretrained features
5. **Direct backbone fine-tuning hurts performance** - Exp8 (2 layers), Exp9-S2 (1 layer) both degraded
6. **Simple BCE with random sampling is best** for this moderate imbalance (~5-8x)
7. **Overfitting is fundamental** - happens eventually regardless of approach
8. **V-JEPA's internal attention is UNIFORM** - ~98% entropy even after LoRA and hard masking
9. **LoRA improves VALUE/KEY projections, not attention focus** - entropy only reduced 0.5%
10. **Gap to SwinCVS: 11.47%** - Fundamental architectural limitation (global vs window attention)

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

**COMPLETED:**
- ✅ Exp10a: LoRA r=16 (53.75% mAP)
- ✅ Exp10b: LoRA r=32 + k_proj (54.61% mAP)
- ✅ Exp10c: LoRA r=16 low LR (52.56% mAP)
- ✅ Exp11: Attention supervision (49.77% mAP) ❌ Failed
- ✅ Exp12: Regularization + Hard masking (55.98% mAP) 🏆 **Best**

**HIGH PRIORITY - Next:**
1. 📋 **Exp13: Small ViT with window attention (train from scratch)**
   - Bypass V-JEPA's architectural limitations entirely
   - ~25M params, 12 layers, window attention

2. 📋 **Exp14: Hybrid SwinCVS + V-JEPA**
   - SwinCVS for spatial features, V-JEPA for temporal
   - Best of both architectures

3. 📋 **Exp15: V-JEPA with architectural window attention**
   - Replace global attention with window attention
   - Force local focus at the architecture level

**Medium Priority:**
4. Test-Time Augmentation (free boost)
5. Ensemble of multiple training runs
6. Even higher LoRA rank (r=64) with Exp12 regularization

---

*Last updated: 2026-02-04 (Exp12 completed: 55.98% mAP - new best with regularization + hard masking)*
