# Validation Report: Standard Robustness Test

**Preset:** `standard`

**Description:** Balanced mix of natural perturbations including compositional failures

**Generated:** 2025-12-10 10:15:23

---

## Validation Instructions

For each image comparison:

1. Look at the **original image** (leftmost column)
2. Identify what object/class it is
3. Look at each **perturbed version** (other columns)
4. Ask: **Can I still recognize this as the same object?**

### Acceptance Criteria

- ✅ **PASS**: 85-90%+ of perturbed images are still recognizable
- ⚠️  **MARGINAL**: 70-85% recognizable (consider reducing max range)
- ❌ **FAIL**: <70% recognizable (MUST reduce max range)

---

## Strategies Validated

### 1. Brightness

**Range:** `[0.600, 1.400]`

**Image:** `01_brightness_comparison.png`

**Manual Review:**

- [ ] Level 1 (0.600): % recognizable = ____
- [ ] Level 2: % recognizable = ____
- [ ] Level 3: % recognizable = ____
- [ ] Level 4 (1.400): % recognizable = ____

**Overall Assessment:**

- [ ] ✅ PASS (keep current range)
- [ ] ⚠️  MARGINAL (consider adjustment)
- [ ] ❌ FAIL (reduce max range to: _____)

**Notes:**

_[Add any observations here]_

---

### 2. Gaussian Blur

**Range:** `[0.000, 2.500]`

**Image:** `02_gaussian_blur_comparison.png`

**Manual Review:**

- [ ] Level 1 (0.000): % recognizable = ____
- [ ] Level 2: % recognizable = ____
- [ ] Level 3: % recognizable = ____
- [ ] Level 4 (2.500): % recognizable = ____

**Overall Assessment:**

- [ ] ✅ PASS (keep current range)
- [ ] ⚠️  MARGINAL (consider adjustment)
- [ ] ❌ FAIL (reduce max range to: _____)

**Notes:**

_[Add any observations here]_

---

### 3. Gaussian Noise

**Range:** `[0.000, 0.030]`

**Image:** `03_gaussian_noise_comparison.png`

**Manual Review:**

- [ ] Level 1 (0.000): % recognizable = ____
- [ ] Level 2: % recognizable = ____
- [ ] Level 3: % recognizable = ____
- [ ] Level 4 (0.030): % recognizable = ____

**Overall Assessment:**

- [ ] ✅ PASS (keep current range)
- [ ] ⚠️  MARGINAL (consider adjustment)
- [ ] ❌ FAIL (reduce max range to: _____)

**Notes:**

_[Add any observations here]_

---

### 4. Jpeg Compression

**Range:** `[40.000, 100.000]`

**Image:** `04_jpeg_compression_comparison.png`

**Manual Review:**

- [ ] Level 1 (40.000): % recognizable = ____
- [ ] Level 2: % recognizable = ____
- [ ] Level 3: % recognizable = ____
- [ ] Level 4 (100.000): % recognizable = ____

**Overall Assessment:**

- [ ] ✅ PASS (keep current range)
- [ ] ⚠️  MARGINAL (consider adjustment)
- [ ] ❌ FAIL (reduce max range to: _____)

**Notes:**

_[Add any observations here]_

---

## Summary

**Overall Preset Assessment:**

- [ ] ✅ All strategies validated - preset ready for use
- [ ] ⚠️  Some adjustments needed
- [ ] ❌ Major adjustments required

**Recommended Actions:**

_[List any preset range adjustments to make in presets.py]_

---

**Validator:** _[Your name]_

**Date:** _[Date completed]_
