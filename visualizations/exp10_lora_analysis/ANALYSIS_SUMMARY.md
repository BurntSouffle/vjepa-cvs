# Exp10 LoRA Model Analysis Summary
## Model: 53.75% mAP - Our Best Result

---

## Overview

This analysis examines whether LoRA fine-tuning changed V-JEPA's internal attention patterns
and whether the model now focuses more on surgical anatomy.

**Checkpoint:** `results/exp10_lora/run_20260201_203545/best_model.pt`
- Epoch: 2
- Best Val mAP: 53.75%
- LoRA applied to: 72 attention modules (query, value projections)
- LoRA rank: 16, alpha: 32

---

## Key Finding 1: Attention Patterns Unchanged

### Entropy Analysis (Lower = More Focused)

| Model | Entropy | Interpretation |
|-------|---------|----------------|
| Baseline V-JEPA | 98.4% | Nearly uniform attention |
| LoRA V-JEPA | 97.9% | Nearly uniform attention |
| **Improvement** | **0.5%** | Minimal change |

**Conclusion:** LoRA did NOT significantly change V-JEPA's internal attention patterns.
The attention remains ~98% uniform (nearly maximum entropy), meaning the model attends
to all spatial-temporal positions almost equally.

### Visualization Observations
- Heatmaps show nearly uniform red across all spatial positions
- No clear "hot spots" focusing on anatomical structures
- Both baseline and LoRA patterns look virtually identical

---

## Key Finding 2: Prediction Improvements Come from Task Heads

Since V-JEPA's internal attention didn't change, the 53.75% mAP improvement must come from:

1. **LoRA adapters** modifying the VALUE projections (not WHERE to attend, but WHAT to output)
2. **Better-trained task heads** (attention pooler, CVS classifier)
3. **The combination** creating better feature transformations for classification

### How This Works

```
V-JEPA Internal Attention: WHERE to look → Still uniform (~98%)
LoRA Value Projections:    WHAT to output → Modified to be more discriminative
Task Heads:                HOW to classify → Trained on surgical domain
```

---

## Key Finding 3: Predictions on Test Samples

### Sample Analysis

| Sample | Type | GT Labels | Predictions | Accuracy |
|--------|------|-----------|-------------|----------|
| 122_35150 | Negative | [0, 0, 0] | [0.02, 0.00, 0.00] | **3/3** |
| 127_19475 | Full CVS | [1, 1, 1] | [0.76, 0.61, 0.43] | 2/3 |
| 129_68650 | C1-only | [1, 0, 0] | [0.29, 0.11, 0.05] | 2/3 |
| 123_37650 | C2-only | [0, 1, 0] | [0.19, 0.15, 0.01] | 2/3 |

### Detailed Breakdown

**122_35150 (Negative):** PERFECT
- All predictions well below threshold (0.5)
- Model correctly identifies absence of all CVS criteria

**127_19475 (Full CVS):** 2/3 Correct
- C1: 0.76 > 0.5 ✓
- C2: 0.61 > 0.5 ✓
- C3: 0.43 < 0.5 ✗ (missed)
- Model recognizes most criteria but C3 just below threshold

**129_68650 (C1-only):** 2/3 Correct
- C1: 0.29 < 0.5 ✗ (should be 1)
- C2: 0.11 < 0.5 ✓
- C3: 0.05 < 0.5 ✓
- Model fails to detect C1 when it's the only positive criterion

**123_37650 (C2-only):** 2/3 Correct
- C1: 0.19 < 0.5 ✓
- C2: 0.15 < 0.5 ✗ (should be 1)
- C3: 0.01 < 0.5 ✓
- Model fails to detect C2 when it's the only positive criterion

### Pattern Observed
- **Strong on negatives** (all criteria absent)
- **Good on full CVS** (all criteria present)
- **Weak on single-criterion** samples (subtle visibility)

---

## Key Finding 4: C2 Detection Analysis

The previous analysis showed 0/5 C2-only correct. Now:

| Sample | C2 GT | C2 Pred | Result |
|--------|-------|---------|--------|
| 123_37650 | 1 | 0.15 | WRONG |

**C2 detection is still challenging.** The Cystic Plate criterion requires subtle
anatomical recognition that the model still struggles with.

---

## Conclusions

### What LoRA Did
1. **Did NOT change attention patterns** - V-JEPA still attends uniformly (~98%)
2. **DID modify feature transformations** - Through value projections
3. **DID improve task performance** - 53.75% mAP vs ~50% baseline

### Why Attention Stayed Uniform
V-JEPA was pretrained on natural videos with self-supervised learning. The attention
mechanism learns to aggregate information across the entire video for general understanding.
LoRA with rank=16 is too small to fundamentally change this behavior.

### Implications for Future Work

1. **Attention uniformity is not necessarily bad**
   - The model extracts global features that the task heads can decode
   - Improvement came from better feature transformation, not focused attention

2. **To change attention patterns, consider:**
   - Higher LoRA rank (32, 64)
   - Training key projections (not just query/value)
   - Explicit attention supervision
   - Architecture modifications

3. **Single-criterion detection needs:**
   - More balanced training data
   - Hard negative mining
   - Criterion-specific auxiliary losses

---

## Files Generated

```
visualizations/exp10_lora_analysis/
├── attention_comparison.png    # Side-by-side baseline vs LoRA attention
├── entropy_analysis.png        # Bar chart of entropy comparison
├── predictions_comparison.png  # Predictions on each sample
└── ANALYSIS_SUMMARY.md         # This summary
```

---

## Technical Details

- **LoRA Configuration:**
  - Rank (r): 16
  - Alpha: 32
  - Target modules: query, value projections
  - Dropout: 0.1

- **Entropy Calculation:**
  ```
  entropy = -Σ(p_i * log(p_i)) / log(N)
  ```
  Where N is the number of tokens (2048 for 8 temporal × 256 spatial).
  100% = perfectly uniform, 0% = single token attended.

- **Samples Analyzed:**
  - 122_35150: Video 122, frame 35150 (Negative)
  - 127_19475: Video 127, frame 19475 (Full CVS)
  - 129_68650: Video 129, frame 68650 (C1-only)
  - 123_37650: Video 123, frame 37650 (C2-only)
