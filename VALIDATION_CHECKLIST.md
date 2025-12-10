# Preset Validation Checklist

**Quick reference for tracking manual validation progress**

---

## ✅ Implementation (COMPLETE)

- [x] Created 4 preset configurations (standard, lighting, blur, corruption)
- [x] Documented each preset with:
  - [x] Description
  - [x] Perturbation ranges with justifications
  - [x] Time estimates
  - [x] Use cases
- [x] Created validation script (`validation/validate_presets.py`)
- [x] Generated 50+ comparison images per preset (1,040 total comparisons)

---

## ⏳ Validation (PENDING MANUAL REVIEW)

### Standard Preset

**Location:** `validation/output/standard/`

- [ ] Review `01_brightness_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL

- [ ] Review `02_gaussian_blur_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL

- [ ] Review `03_gaussian_noise_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL

- [ ] Review `04_jpeg_compression_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL

- [ ] Fill out `VALIDATION_REPORT.md`
- [ ] Calculate overall rate: ____%
- [ ] If <85%, adjust ranges in `presets.py` and re-run
- [ ] Update `VALIDATION_STATUS['standard']` when complete

**Overall Status:** ⬜ PASS (≥87%) / ⬜ MARGINAL (85-87%) / ⬜ FAIL (<85%)

---

### Lighting Preset

**Location:** `validation/output/lighting/`

- [ ] Review `01_brightness_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL

- [ ] Review `02_contrast_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL

- [ ] Review `03_gamma_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL

- [ ] Fill out `VALIDATION_REPORT.md`
- [ ] Calculate overall rate: ____%
- [ ] If <85%, adjust ranges in `presets.py` and re-run
- [ ] Update `VALIDATION_STATUS['lighting']` when complete

**Overall Status:** ⬜ PASS (≥85%) / ⬜ MARGINAL (82-85%) / ⬜ FAIL (<82%)

---

### Blur Preset

**Location:** `validation/output/blur/`

- [ ] Review `01_gaussian_blur_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL

- [ ] Review `02_motion_blur_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL
  - [ ] **⚠️ If <80%:** Reduce max_kernel from 25 to 20

- [ ] Review `03_jpeg_compression_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL
  - [ ] **⚠️ If <80%:** Increase min_quality from 30 to 40

- [ ] Fill out `VALIDATION_REPORT.md`
- [ ] Calculate overall rate: ____%
- [ ] If <85%, adjust ranges in `presets.py` and re-run
- [ ] Update `VALIDATION_STATUS['blur']` when complete

**Overall Status:** ⬜ PASS (≥85%) / ⬜ MARGINAL (80-85%) / ⬜ FAIL (<80%)

---

### Corruption Preset

**Location:** `validation/output/corruption/`

- [ ] Review `01_gaussian_noise_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL

- [ ] Review `02_jpeg_compression_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL
  - [ ] **⚠️ LIKELY FAIL:** Quality 10 is extreme
  - [ ] **If <75%:** Increase min_quality from 10 to 20

- [ ] Review `03_gaussian_blur_comparison.png` (20 samples × 4 levels)
  - [ ] Count recognizable at level 4 (max): ____/20 = ____%
  - [ ] Target: ≥85% → Status: ⬜ PASS / ⬜ FAIL

- [ ] Fill out `VALIDATION_REPORT.md`
- [ ] Calculate overall rate: ____%
- [ ] If <80%, adjust ranges in `presets.py` and re-run
- [ ] Update `VALIDATION_STATUS['corruption']` when complete

**Overall Status:** ⬜ PASS (≥85%) / ⬜ MARGINAL (80-85%) / ⬜ FAIL (<80%)

**Note:** This is the most aggressive preset, 80-85% overall is acceptable.

---

## After Manual Review

- [ ] All presets validated with ≥80-90% label preservation
- [ ] All `VALIDATION_REPORT.md` files filled out
- [ ] Updated `VALIDATION_STATUS` in `src/visprobe/presets.py` for all presets
- [ ] Tested time estimates with real model run:
  ```bash
  python examples/preset_comparison.py
  ```
- [ ] Time estimates are accurate (within ±20%)
- [ ] Committed validation results:
  ```bash
  git add validation/output/
  git add src/visprobe/presets.py
  git add PRESET_DESIGN.md
  git add VALIDATION_SUMMARY.md
  git add VALIDATION_CHECKLIST.md
  git commit -m "Complete preset validation with manual review

  - Generated 1,040 comparison images across 4 presets
  - Manually validated all perturbation ranges
  - Label preservation rates: [add your rates here]
  - All presets ready for production use"
  ```

---

## Success Criteria (From Original Checklist)

- [x] **4 presets implemented** ✅
- [ ] **Manual validation shows ~85-90% label preservation** ⏳ PENDING
- [x] **Clear documentation of ranges and justifications** ✅ (See PRESET_DESIGN.md)
- [ ] **Time estimates are accurate** ⏳ PENDING (test on real models)

**Overall Status:** 2/4 complete, 2/4 pending manual work

---

## Estimated Time

- **Manual review:** 30-40 minutes
- **Filling out reports:** 15-20 minutes
- **Range adjustments (if needed):** 10-30 minutes
- **Time estimate testing:** 15-20 minutes

**Total:** ~1.5-2 hours to complete validation

---

## Quick Reference: How to Count

1. Open a comparison image
2. Look at the **rightmost column** (level 4, maximum severity)
3. For each of the 20 rows:
   - Compare to the **original** (leftmost)
   - Ask: "Can I still tell what object this is?"
   - ✅ = recognizable, ❌ = unrecognizable
4. Count the checkmarks
5. Calculate: `(✅ / 20) × 100%`

**Example:**
- 18 out of 20 recognizable at max severity
- 18/20 = 90% → **PASS** ✅

**Threshold:**
- ≥85% = PASS ✅
- 80-84% = MARGINAL ⚠️
- <80% = FAIL ❌ (adjust range)

---

## Need Help?

- **Detailed guidance:** See `VALIDATION_SUMMARY.md`
- **Range justifications:** See `PRESET_DESIGN.md`
- **How to run validation:** See `validation/README.md`
- **Adjusting ranges:** Edit `src/visprobe/presets.py`, re-run validation script

---

**Current Status:** ⏳ Ready for manual review

**Blocker:** Human validation required

**Next Action:** Open `validation/output/standard/01_brightness_comparison.png` and start counting!
