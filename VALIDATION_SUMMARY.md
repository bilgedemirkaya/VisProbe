# VisProbe Preset Validation Summary

**Status:** Comparison images generated, awaiting manual review

**Generated:** 2024-12-10

---

## Overview

This document tracks the validation status for all VisProbe presets. The goal is to ensure that perturbed images at maximum severity still preserve the original label (85-90% recognition rate).

---

## Validation Status

| Preset | Images Generated | Strategies Tested | Status | Date | Notes |
|--------|------------------|-------------------|--------|------|-------|
| **standard** | ✅ 4 images | 4/6 (skipped compositional) | ⏳ Pending Review | - | Review validation/output/standard/ |
| **lighting** | ✅ 3 images | 3/4 (skipped compositional) | ⏳ Pending Review | - | Review validation/output/lighting/ |
| **blur** | ✅ 3 images | 3/4 (skipped compositional) | ⏳ Pending Review | - | Review validation/output/blur/ |
| **corruption** | ✅ 3 images | 3/4 (skipped compositional) | ⏳ Pending Review | - | Review validation/output/corruption/ |

**Total Generated:** 13 comparison images × 20 samples × 4 severity levels = **1,040 image comparisons**

---

## Validation Workflow

### Step 1: Review Generated Images ⏳ IN PROGRESS

**Location:** `validation/output/[preset_name]/`

**For each preset:**

1. Open the output directory
2. View each comparison image:
   - **Left column:** Original image
   - **Columns 2-5:** Perturbed versions at increasing severity
3. For each perturbed image, ask: **"Can I still recognize this as the same object/class?"**

### Step 2: Count Recognition Rates

**For each perturbation strategy:**

1. Count how many of the 20 samples are still recognizable at each severity level
2. Calculate: `(recognizable / 20) × 100%`
3. Record in the preset's `VALIDATION_REPORT.md`

**Example:**
```
Brightness (max=1.5):
- Level 1 (0.50): 20/20 = 100% ✅
- Level 2 (0.77): 20/20 = 100% ✅
- Level 3 (1.03): 19/20 = 95% ✅
- Level 4 (1.50): 17/20 = 85% ✅ (borderline, acceptable)

Overall: 85% at max severity → PASS ✅
```

### Step 3: Fill Out Validation Reports

**Location:** `validation/output/[preset_name]/VALIDATION_REPORT.md`

For each strategy:
- [ ] Record recognition rates for all 4 severity levels
- [ ] Mark overall assessment (PASS/MARGINAL/FAIL)
- [ ] Add notes about any edge cases
- [ ] Recommend range adjustments if needed

### Step 4: Adjust Ranges (If Needed)

**If any strategy has <85% recognition at max severity:**

1. Edit `src/visprobe/presets.py`
2. Reduce the `max_*` parameter by 10-20%
3. Re-run validation: `python validation/validate_presets.py --preset [name]`
4. Re-review until target met

### Step 5: Update VALIDATION_STATUS

**Once all strategies pass (≥85% recognition):**

Edit `src/visprobe/presets.py`:

```python
VALIDATION_STATUS = {
    "standard": {
        "validated": True,                    # ← Change to True
        "validation_date": "2024-12-10",     # ← Today's date
        "label_preservation_rate": 0.87,     # ← Your measured rate
        "notes": "Validated with 20 CIFAR-10 samples, all strategies passed",
    },
    # ... repeat for other presets
}
```

---

## Detailed Preset Status

### 1. Standard Preset

**Strategies Validated:**
- ✅ Brightness (0.6 - 1.4)
- ✅ Gaussian Blur (σ: 0.0 - 2.5)
- ✅ Gaussian Noise (std: 0.0 - 0.03)
- ✅ JPEG Compression (quality: 40 - 100)
- ⏭️ Compositional: low_light_blur (skipped - validate components first)
- ⏭️ Compositional: compressed_noisy (skipped - validate components first)

**Comparison Images:**
- `01_brightness_comparison.png` - 20 samples × 4 levels
- `02_gaussian_blur_comparison.png` - 20 samples × 4 levels
- `03_gaussian_noise_comparison.png` - 20 samples × 4 levels
- `04_jpeg_compression_comparison.png` - 20 samples × 4 levels

**Manual Review Checklist:**
- [ ] Reviewed brightness images
- [ ] Reviewed gaussian_blur images
- [ ] Reviewed gaussian_noise images
- [ ] Reviewed jpeg_compression images
- [ ] Filled out VALIDATION_REPORT.md
- [ ] All strategies pass (≥85% recognition)
- [ ] Updated VALIDATION_STATUS in presets.py

**Expected Results:**
- Brightness: ~90% (conservative range)
- Gaussian Blur: ~85-88% (moderate challenge)
- Gaussian Noise: ~92% (very conservative)
- JPEG Compression: ~88% (moderate challenge)
- **Overall target: 87-90%**

---

### 2. Lighting Preset

**Strategies Validated:**
- ✅ Brightness (0.5 - 1.5)
- ✅ Contrast (0.7 - 1.3)
- ✅ Gamma (0.7 - 1.3)
- ⏭️ Compositional: dim_low_contrast (skipped)

**Comparison Images:**
- `01_brightness_comparison.png` - 20 samples × 4 levels
- `02_contrast_comparison.png` - 20 samples × 4 levels
- `03_gamma_comparison.png` - 20 samples × 4 levels

**Manual Review Checklist:**
- [ ] Reviewed brightness images
- [ ] Reviewed contrast images
- [ ] Reviewed gamma images
- [ ] Filled out VALIDATION_REPORT.md
- [ ] All strategies pass (≥85% recognition)
- [ ] Updated VALIDATION_STATUS in presets.py

**Expected Results:**
- Brightness: ~85% (more aggressive than standard)
- Contrast: ~90% (conservative)
- Gamma: ~88% (moderate)
- **Overall target: 85-88%**

---

### 3. Blur Preset

**Strategies Validated:**
- ✅ Gaussian Blur (σ: 0.0 - 3.0)
- ✅ Motion Blur (kernel: 1 - 25)
- ✅ JPEG Compression (quality: 30 - 100)
- ⏭️ Compositional: motion_compressed (skipped)

**Comparison Images:**
- `01_gaussian_blur_comparison.png` - 20 samples × 4 levels
- `02_motion_blur_comparison.png` - 20 samples × 4 levels
- `03_jpeg_compression_comparison.png` - 20 samples × 4 levels

**Manual Review Checklist:**
- [ ] Reviewed gaussian_blur images
- [ ] Reviewed motion_blur images
- [ ] Reviewed jpeg_compression images
- [ ] Filled out VALIDATION_REPORT.md
- [ ] All strategies pass (≥85% recognition)
- [ ] Updated VALIDATION_STATUS in presets.py

**Expected Results:**
- Gaussian Blur: ~85% (σ=3.0 is significant blur)
- Motion Blur: ~82-85% (kernel=25 is aggressive, may need tuning)
- JPEG Compression: ~85% (quality 30 is harsh)
- **Overall target: 83-87%** (most aggressive preset)

**⚠️ Potential Adjustments:**
- Motion blur max may need reduction: 25 → 20
- JPEG quality min may need increase: 30 → 40

---

### 4. Corruption Preset

**Strategies Validated:**
- ✅ Gaussian Noise (std: 0.0 - 0.05)
- ✅ JPEG Compression (quality: 10 - 100)
- ✅ Gaussian Blur (σ: 0.0 - 2.0)
- ⏭️ Compositional: degraded_transmission (skipped)

**Comparison Images:**
- `01_gaussian_noise_comparison.png` - 20 samples × 4 levels
- `02_jpeg_compression_comparison.png` - 20 samples × 4 levels
- `03_gaussian_blur_comparison.png` - 20 samples × 4 levels

**Manual Review Checklist:**
- [ ] Reviewed gaussian_noise images
- [ ] Reviewed jpeg_compression images
- [ ] Reviewed gaussian_blur images
- [ ] Filled out VALIDATION_REPORT.md
- [ ] All strategies pass (≥85% recognition)
- [ ] Updated VALIDATION_STATUS in presets.py

**Expected Results:**
- Gaussian Noise: ~85% (std=0.05 is very noisy)
- JPEG Compression: ~80-83% (quality 10 is EXTREME, likely needs adjustment)
- Gaussian Blur: ~90% (σ=2.0 is conservative)
- **Overall target: 80-85%**

**⚠️ Potential Adjustments:**
- JPEG quality min **VERY LIKELY** needs increase: 10 → 20 or 30
- This is the most aggressive preset, some failures expected

---

## Validation Criteria Reference

### Recognition Rate Thresholds

| Rate | Assessment | Action | Example |
|------|------------|--------|---------|
| **90-100%** | ✅ EXCELLENT | Keep range, possibly increase max slightly | Contrast 0.7-1.3 |
| **85-89%** | ✅ PASS | Keep range, no changes needed | Target zone |
| **80-84%** | ⚠️ MARGINAL | Consider reducing max by 10% | Motion blur 25 → 22 |
| **75-79%** | ⚠️ NEEDS ADJUSTMENT | Reduce max by 15-20% | JPEG 10 → 15 |
| **<75%** | ❌ FAIL | Reduce max by 20-30% and re-validate | JPEG 10 → 20 |

### How to Count

**For each severity level in an image:**

1. Look at all 20 sample rows
2. For each row, compare original (left) to perturbed (one of the 4 levels)
3. Count samples where you can still confidently identify the object class
4. Calculate: `(recognizable / 20) × 100%`

**Example:** Brightness at max (1.5x)
- Row 1 (airplane): Still clearly an airplane ✅
- Row 2 (automobile): Still clearly a car ✅
- Row 3 (bird): Too bright, can't tell ❌
- Row 4 (cat): Still see cat features ✅
- ... (count all 20)
- **Result: 17/20 = 85%** → PASS (borderline)

### Edge Cases

**What counts as "recognizable"?**
- ✅ You can identify the broad category (animal, vehicle, object)
- ✅ Colors are shifted but shape is clear
- ✅ Details are lost but main features visible
- ❌ Could be multiple different objects
- ❌ Just a blob/noise with no clear features
- ❌ You can only tell because you saw the original

**When in doubt:** Mark as unrecognizable (be conservative)

---

## Common Issues and Solutions

### Issue 1: "Motion blur at max is too aggressive"

**Symptoms:** <80% recognizable at kernel=25
**Solution:**
```python
# In presets.py, line 79
{"type": "motion_blur", "min_kernel": 1, "max_kernel": 20, "angle": 0},  # was 25
```
**Re-validate:** `python validation/validate_presets.py --preset blur`

### Issue 2: "JPEG quality 10 is unrecognizable"

**Symptoms:** <75% recognizable at quality=10
**Solution:**
```python
# In presets.py, line 100
{"type": "jpeg_compression", "min_quality": 20, "max_quality": 100},  # was 10
```
**Re-validate:** `python validation/validate_presets.py --preset corruption`

### Issue 3: "Gamma looks weird/wrong"

**Possible cause:** Gamma correction may look unusual on different displays
**Solution:** View on a calibrated monitor, or trust that values within 0.7-1.3 are standard

### Issue 4: "All images are too easy, >95% pass"

**This is OK!** Conservative ranges are safe. Only increase if:
- You want more challenging tests
- You've verified the model is too robust
- You're comfortable with potential false failures

---

## Timeline

### Completed ✅

- [x] **2024-12-10:** Created preset configurations
- [x] **2024-12-10:** Implemented 4 presets (standard, lighting, blur, corruption)
- [x] **2024-12-10:** Documented ranges in PRESET_DESIGN.md
- [x] **2024-12-10:** Created validation script (validate_presets.py)
- [x] **2024-12-10:** Generated 13 comparison images (1,040 comparisons total)

### Pending ⏳

- [ ] **Manual review** of all 13 comparison images
- [ ] **Fill out** 4 VALIDATION_REPORT.md files
- [ ] **Tune ranges** if any strategies fail (<85%)
- [ ] **Re-validate** any adjusted ranges
- [ ] **Update** VALIDATION_STATUS in presets.py
- [ ] **Commit** validation results and reports

### Estimated Time for Manual Review

- **Per image:** 2-3 minutes (20 samples × 4 levels = 80 comparisons)
- **Per preset:** 6-10 minutes (3-4 images)
- **Total:** 30-40 minutes for all 4 presets

**Recommended approach:**
1. Do one preset at a time
2. Take breaks between presets
3. Get a second opinion on edge cases
4. Use the VALIDATION_REPORT.md template to stay organized

---

## Next Steps

### Immediate (YOU - Manual Reviewer)

1. **Open** `validation/output/standard/` in file browser
2. **Review** each PNG image carefully
3. **Count** recognizable images at each severity level
4. **Fill out** `VALIDATION_REPORT.md` for standard preset
5. **Repeat** for lighting, blur, corruption presets
6. **Adjust** ranges in presets.py if needed
7. **Update** VALIDATION_STATUS when complete

### After Validation Complete

- [ ] Commit changes:
  ```bash
  git add src/visprobe/presets.py
  git add validation/output/
  git add VALIDATION_SUMMARY.md
  git commit -m "Complete preset validation with manual review"
  ```

- [ ] Update documentation:
  - [ ] Add validation results to README
  - [ ] Update preset descriptions if ranges changed
  - [ ] Document final label preservation rates

- [ ] Test with real model:
  ```bash
  python examples/preset_comparison.py
  ```
  Verify that time estimates are accurate

---

## Validation Checklist

**Before marking validation as complete:**

- [ ] Generated comparison images for all 4 presets (✅ DONE)
- [ ] Manually reviewed all images (⏳ PENDING)
- [ ] Calculated recognition rates for each strategy (⏳ PENDING)
- [ ] Filled out all VALIDATION_REPORT.md files (⏳ PENDING)
- [ ] Made necessary range adjustments (⏳ IF NEEDED)
- [ ] Re-validated any changes (⏳ IF NEEDED)
- [ ] Overall preservation rates meet targets:
  - [ ] Standard: 87-90%
  - [ ] Lighting: 85-88%
  - [ ] Blur: 83-87%
  - [ ] Corruption: 80-85%
- [ ] Updated VALIDATION_STATUS in presets.py (⏳ PENDING)
- [ ] Tested time estimates with real model (⏳ PENDING)
- [ ] Committed all validation artifacts (⏳ PENDING)

**Once all boxes checked:** Presets are production-ready ✅

---

## Questions or Issues?

- **Can't decide if an image is recognizable:** Get a second opinion, err on side of caution
- **Recognition rate is borderline (83-84%):** Your call - either keep or reduce max slightly
- **Confused about a strategy:** See PRESET_DESIGN.md for rationale
- **Script errors:** Check that CIFAR-10 is downloaded, PyTorch installed

---

**Current Status:** ⏳ Images generated, ready for manual review

**Blocker:** Needs human validation to proceed

**Owner:** YOU (manual reviewer)

**Estimated completion:** 30-40 minutes of focused review
