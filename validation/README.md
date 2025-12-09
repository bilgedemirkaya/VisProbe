# Preset Validation

This directory contains tools for validating VisProbe preset configurations.

## Why Validation?

The preset parameter ranges (e.g., brightness from 0.5 to 1.5) must be carefully chosen to ensure that perturbed images **still preserve the original label**. If the perturbations are too severe, a perturbed image of a "cat" might look like noise, making it an unfair test.

**Goal:** 85-90% of perturbed images should still be recognizable by a human as the original class.

---

## Quick Start

### 1. Run Validation Script

```bash
# Validate a specific preset
python validation/validate_presets.py --preset lighting

# Validate all presets
python validation/validate_presets.py --all

# Use more samples for thorough validation
python validation/validate_presets.py --preset standard --num-samples 20
```

### 2. Review Generated Images

Open `validation/output/[preset_name]/` and look at the comparison images.

For each image:
- **Left column**: Original image
- **Other columns**: Perturbed versions at different severity levels

### 3. Fill Out Validation Report

Open `validation/output/[preset_name]/VALIDATION_REPORT.md` and:
- Count how many images are still recognizable at each level
- Mark PASS/MARGINAL/FAIL for each strategy
- Note any recommended range adjustments

### 4. Update Preset Ranges

If adjustments are needed:

1. Edit `src/visprobe/presets.py`
2. Adjust `min_X` and `max_X` values
3. Re-run validation to verify

### 5. Mark as Validated

Once satisfied, update `src/visprobe/presets.py`:

```python
VALIDATION_STATUS = {
    "lighting": {
        "validated": True,                    # ← Change to True
        "validation_date": "2024-12-09",     # ← Add date
        "label_preservation_rate": 0.87,     # ← Add your measured rate
        "notes": "Validated with 20 samples",
    },
    # ...
}
```

---

## Validation Criteria

### Acceptance Levels

| Recognition Rate | Assessment | Action |
|-----------------|------------|--------|
| **85-100%** | ✅ PASS | Keep current range |
| **70-85%** | ⚠️ MARGINAL | Consider reducing max |
| **< 70%** | ❌ FAIL | Must reduce max range |

### How to Count

For each perturbation severity level:

1. Look at all test images (e.g., 10 samples)
2. Count how many are **still clearly recognizable** as the original class
3. Calculate: `(recognizable / total) * 100%`

**Example:**
- 10 images tested at max brightness (1.5x)
- 9 are still clearly recognizable as their class
- 1 is too bright to identify
- **Recognition rate: 90%** ✅ PASS

---

## Example Validation Session

```bash
$ python validation/validate_presets.py --preset lighting --num-samples 10

======================================================================
Validating Preset: LIGHTING
======================================================================

Preset: Lighting Conditions
Description: Tests robustness to brightness, contrast, and gamma variations

Loading 10 CIFAR-10 test samples...
  ✓ Loaded 10 samples

Validating 4 strategies...

  1. Validating: brightness
     Range: [0.500, 1.500]
  ✓ Saved: validation/output/lighting/01_brightness_comparison.png

  2. Validating: contrast
     Range: [0.700, 1.300]
  ✓ Saved: validation/output/lighting/02_contrast_comparison.png

  ...

======================================================================
✅ Validation complete for 'lighting'
======================================================================

📁 Output directory: validation/output/lighting
📄 Validation report: validation/output/lighting/VALIDATION_REPORT.md

👁️  NEXT STEPS:
   1. Open the output directory and review all images
   2. For each perturbation, ask: 'Can I still recognize the object?'
   3. If >15% of images are unrecognizable, reduce the max range
   4. Update ranges in src/visprobe/presets.py
   5. Re-run validation to verify improvements

   Target: 85-90% of perturbed images should be recognizable
```

---

## Validation Workflow

```
┌─────────────────────────────────────┐
│ 1. Run validation script            │
│    python validation/validate_...   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. Review generated images          │
│    Look at comparison grids         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. Fill out VALIDATION_REPORT.md    │
│    Count recognizable images        │
└──────────────┬──────────────────────┘
               │
               ▼
          ┌────┴────┐
          │ Passed? │
          └────┬────┘
               │
        ┌──────┴──────┐
        │             │
       Yes            No
        │             │
        │             ▼
        │    ┌────────────────────┐
        │    │ 4a. Adjust ranges  │
        │    │     in presets.py  │
        │    └────────┬───────────┘
        │             │
        │             ▼
        │    ┌────────────────────┐
        │    │ 4b. Re-run         │
        │    │     validation     │
        │    └────────┬───────────┘
        │             │
        └─────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 5. Mark as validated in presets.py │
│    Update VALIDATION_STATUS         │
└─────────────────────────────────────┘
```

---

## Tips for Manual Review

### Good Examples (Keep Range)

**Brightness at 1.5x:**
- Image is brighter but object is clearly visible
- Colors may be washed out but shape is recognizable
- You can confidently identify the class

**Blur at sigma=2.0:**
- Image is blurry but main features visible
- You can still tell it's a cat/car/etc.
- Just slightly harder to see details

### Bad Examples (Reduce Range)

**Brightness at 2.0x:**
- Image is completely washed out
- Can't tell what object it is
- Just looks like white noise

**Blur at sigma=5.0:**
- Image is so blurry it's unrecognizable
- Can't distinguish object from background
- Could be anything

### Edge Cases

If you're unsure whether an image is recognizable:
- Ask: "If I didn't know the label, could I identify this?"
- Show it to a colleague without context
- Err on the side of caution (mark as unrecognizable)

---

## Common Range Adjustments

Based on typical validation results:

| Perturbation | Initial Range | Common Adjustment | Final Range |
|--------------|---------------|-------------------|-------------|
| Brightness | [0.5, 1.5] | None needed | [0.5, 1.5] ✅ |
| Contrast | [0.7, 1.3] | None needed | [0.7, 1.3] ✅ |
| Gamma | [0.7, 1.3] | Reduce max | [0.7, 1.2] |
| Gaussian Blur | [0, 2.5] | Reduce max | [0, 2.0] |
| Motion Blur | [1, 25] | Reduce max | [1, 20] |
| JPEG Quality | [10, 100] | Increase min | [20, 100] |
| Noise | [0, 0.05] | None needed | [0, 0.05] ✅ |

---

## Troubleshooting

### "Images look strange/wrong"
- Check that you're viewing the comparison images, not raw tensors
- Images should be in normal RGB format
- If colors are off, there may be a normalization issue

### "All images are unrecognizable"
- The ranges are probably too aggressive
- Start with more conservative ranges
- Gradually increase to find the sweet spot

### "Can't tell if edge cases are recognizable"
- Use more samples (`--num-samples 20`)
- Get a second opinion from a colleague
- When in doubt, be conservative (reduce range)

---

## Output Structure

After running validation:

```
validation/
├── README.md                        # This file
├── validate_presets.py              # Validation script
└── output/
    ├── lighting/
    │   ├── 01_brightness_comparison.png
    │   ├── 02_contrast_comparison.png
    │   ├── 03_gamma_comparison.png
    │   ├── 04_dim_low_contrast_comparison.png
    │   └── VALIDATION_REPORT.md
    ├── standard/
    │   ├── 01_brightness_comparison.png
    │   ├── ...
    │   └── VALIDATION_REPORT.md
    ├── blur/
    │   └── ...
    └── corruption/
        └── ...
```

---

## Questions?

- See `../README.md` for general VisProbe documentation
- See `../src/visprobe/presets.py` for preset definitions
- Open an issue if you find validation issues

---

## Validation Checklist

Before marking a preset as validated:

- [ ] Generated comparison images for all strategies
- [ ] Manually reviewed all images
- [ ] Calculated recognition rates for each level
- [ ] Filled out VALIDATION_REPORT.md
- [ ] Made any necessary range adjustments
- [ ] Re-validated if changes were made
- [ ] Updated VALIDATION_STATUS in presets.py
- [ ] Committed changes with validation results

**Once complete, the preset is ready for production use!** ✅
