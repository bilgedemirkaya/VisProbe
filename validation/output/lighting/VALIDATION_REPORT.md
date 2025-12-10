# Validation Report: Lighting Conditions

**Preset:** `lighting`

**Description:** Tests robustness to brightness, contrast, and gamma variations

**Generated:** 2025-12-10 10:16:13

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

**Range:** `[0.500, 1.500]`

**Image:** `01_brightness_comparison.png`

**Manual Review:**

- [ ] Level 1 (0.500): % recognizable = ____
- [ ] Level 2: % recognizable = ____
- [ ] Level 3: % recognizable = ____
- [ ] Level 4 (1.500): % recognizable = ____

**Overall Assessment:**

- [ ] ✅ PASS (keep current range)
- [ ] ⚠️  MARGINAL (consider adjustment)
- [ ] ❌ FAIL (reduce max range to: _____)

**Notes:**

_[Add any observations here]_

---

### 2. Contrast

**Range:** `[0.700, 1.300]`

**Image:** `02_contrast_comparison.png`

**Manual Review:**

- [ ] Level 1 (0.700): % recognizable = ____
- [ ] Level 2: % recognizable = ____
- [ ] Level 3: % recognizable = ____
- [ ] Level 4 (1.300): % recognizable = ____

**Overall Assessment:**

- [ ] ✅ PASS (keep current range)
- [ ] ⚠️  MARGINAL (consider adjustment)
- [ ] ❌ FAIL (reduce max range to: _____)

**Notes:**

_[Add any observations here]_

---

### 3. Gamma

**Range:** `[0.700, 1.300]`

**Image:** `03_gamma_comparison.png`

**Manual Review:**

- [ ] Level 1 (0.700): % recognizable = ____
- [ ] Level 2: % recognizable = ____
- [ ] Level 3: % recognizable = ____
- [ ] Level 4 (1.300): % recognizable = ____

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
