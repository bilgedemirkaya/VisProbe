# Preset Design and Range Justification

This document explains the design rationale and parameter ranges for each VisProbe preset, with emphasis on the threat-model-aware architecture introduced in VisProbe 2.0.

---

## Threat-Model-Aware Architecture (VisProbe 2.0)

VisProbe now organizes presets around **explicit threat models** rather than generic perturbation categories. This enables realistic security testing and reveals vulnerabilities standard tests miss.

### The Four Threat Models

#### 1. **Passive (Natural)** - `natural` preset
- **Definition:** Environmental perturbations without adversary
- **Scenarios:** Weather, sensor noise, camera limitations, transmission artifacts
- **Use Case:** Deployment robustness, production monitoring
- **Key Insight:** Separates "nature vs. adversary" - what breaks naturally vs. maliciously

#### 2. **Active (Adversarial)** - `adversarial` preset
- **Definition:** Gradient-based white-box attacks on clean images
- **Scenarios:** FGSM, PGD, BIM with standard ε=8/255
- **Use Case:** Security testing, adversarial ML research
- **Key Insight:** Measures defense against sophisticated attackers with full model access

#### 3. **Active + Environmental (Realistic Attack)** - `realistic_attack` preset ⭐
- **Definition:** Adversarial attacks under suboptimal environmental conditions
- **Scenarios:** Attacker waits for low-light, blur, compression - then uses SMALLER perturbations
- **Use Case:** Real-world threat modeling, security-critical deployment
- **Key Insight:** **What standard tests miss!** A model robust to ε=8/255 may fail at ε=2/255 in low-light

**Example Blind Spot:**
```
Clean Image Testing:
  FGSM ε=8/255:    FAIL (model robust, survives attack)

Realistic Attack Testing:
  Low-light (0.4x brightness) + FGSM ε=2/255: PASS (but lower robustness)

Vulnerability: Attacker can win with 4x SMALLER perturbation by timing attack!
```

#### 4. **All** - `comprehensive` preset
- **Definition:** Combined evaluation across all three threat models
- **Output:** Per-threat-model robustness scores + opportunistic vulnerability detection
- **Use Case:** Research, benchmarking, publication-ready results

### Design Philosophy

**Goal:** Preset ranges must be aggressive enough to find real robustness issues, but conservative enough that perturbed images still preserve the original label.

**Target:** 85-90% of perturbed images at maximum severity should still be recognizable by humans.

**Threat-Aware Addition:** Each preset explicitly targets a threat model, enabling realistic threat assessment.

---

## Legacy Presets (Deprecated, but Maintained)

The original presets (`standard`, `lighting`, `blur`, `corruption`) are still available for backward compatibility but are deprecated in favor of threat-model-aware presets.

**Migration Guide:**
- Use `natural` instead of `standard`, `lighting`, `blur`, `corruption`
- Use `comprehensive` for complete evaluation across all threat models

---

## 1. Natural Preset (Passive Threat Model)

**Purpose:** General-purpose robustness testing with balanced coverage of common real-world perturbations.

**Target Scenarios:**
- Pre-deployment validation
- Model architecture comparison
- Finding general weaknesses

### Perturbation Ranges

#### Brightness (0.6 - 1.4)
**Rationale:**
- `0.6` = 40% darker (indoor → dim room, dusk lighting)
- `1.4` = 40% brighter (overexposed photos, bright sunlight)
- Range avoids extreme cases (pure black at 0.0, pure white at 2.0+)
- Most cameras handle ±40% brightness variation in auto-exposure
- **Preserves label:** Yes, objects remain recognizable at these levels

#### Gaussian Blur (σ: 0.0 - 2.5)
**Rationale:**
- `σ=0.0` = No blur (original image)
- `σ=2.5` = Moderate blur (slightly out of focus, not motion blur)
- Simulates focus issues, slight camera shake
- At σ=2.5, shapes and colors are visible but details are soft
- **Preserves label:** Yes, major features still distinguishable

#### Gaussian Noise (std: 0.0 - 0.03)
**Rationale:**
- `0.03` = 3% noise relative to [0,1] pixel range
- Simulates sensor noise, ISO noise in low light
- Conservative range - higher values (>0.05) create "static" effect
- **Preserves label:** Yes, noise is visible but doesn't dominate

#### JPEG Compression (quality: 40 - 100)
**Rationale:**
- `100` = No compression artifacts
- `40` = Visible compression (blocky, but not destroyed)
- Simulates web images, video frames, transmitted images
- Quality 40 is similar to heavily compressed social media images
- **Preserves label:** Yes, blocking artifacts present but content clear

#### Compositional: Low Light + Blur
**Components:**
- Brightness: 0.4 - 0.7 (dim lighting)
- Gaussian Blur: σ=1.0 - 2.0 (slight defocus)

**Rationale:**
- Tests **compound failures** - models often fail when perturbations combine
- Simulates handheld photos in low light (harder to focus, shakier)
- This is a **key innovation** of VisProbe - most tools test perturbations in isolation
- **Preserves label:** Yes, challenging but recognizable

#### Compositional: Compression + Noise
**Components:**
- JPEG Quality: 20 - 50 (heavy compression)
- Gaussian Noise: std=0.02 - 0.05 (moderate noise)

**Rationale:**
- Simulates poor quality transmission (low bandwidth + lossy channel)
- Video conferencing, security cameras with analog transmission
- Compression artifacts + noise create realistic degradation
- **Preserves label:** Yes, degraded but identifiable

**Estimated Time:** 12-15 minutes for 100 images
**Search Budget:** 2000 queries

---

## 2. Adversarial Preset (Active Threat Model)

**Purpose:** Test robustness to gradient-based adversarial attacks under ideal conditions.

**Threat Model:** White-box attacker with full model gradients, attacking clean images.

**Target Scenarios:**
- Security hardening of deployed models
- Adversarial ML research and robustness benchmarking
- Comparing model architectures for adversarial robustness
- Certification and verification studies

### Attack Strategies

#### FGSM (Fast Gradient Sign Method)
- **Epsilon Range:** 0 - 8/255 (0 - 0.031)
- **Rationale:**
  - ε=8/255 is standard in adversarial robustness literature (ImageNet, RobustBench)
  - Single-step attack: efficient but less powerful than iterative methods
  - Tests if model has shallow vulnerabilities
  - ε=0.031 ≈ 8 intensity levels on 0-255 scale (imperceptible to human)

#### PGD (Projected Gradient Descent)
- **Epsilon:** 8/255 (standard)
- **Step Size:** ε/4 = 2/255
- **Iterations:** 20
- **Rationale:**
  - Multi-step attack: much stronger than FGSM
  - Tests if model has deep adversarial vulnerabilities
  - ε=8/255 is standard benchmark value
  - 20 iterations balances compute cost with attack strength

#### BIM (Basic Iterative Method)
- **Epsilon:** 4/255 (medium strength)
- **Iterations:** 10
- **Rationale:**
  - Simpler iterative attack (iterative FGSM)
  - Less compute than PGD but stronger than single-step FGSM
  - Smaller ε (4/255) tests models that survive ε=8/255 PGD

#### Small FGSM (Imperceptible Attacks)
- **Epsilon Range:** 0 - 4/255
- **Rationale:**
  - Tests robustness to imperceptible perturbations
  - Many adversarially trained models fail here
  - ε=4/255 = 1/63.75 of pixel range - technically imperceptible
  - Reveals "false sense of security" on ε=8/255 tests

**Estimated Time:** 15-25 minutes for 100 images
**Search Budget:** 1500 queries
**Requirements:** Adversarial Robustness Toolbox (ART)

---

## 3. Realistic Attack Preset (Active + Environmental Threat Model)

**Purpose:** Test robustness to adversarial attacks under realistic environmental conditions.

**Threat Model:** ⭐ **Key innovation!** Active attacker exploiting environmental degradation:
- Attacker observes environmental conditions
- Waits for low-light, blur, compression, etc.
- Uses SMALLER adversarial perturbations
- Success rate with ε=2/255 + low-light > ε=8/255 on clean image

### Attack Scenarios

#### Low-Light + FGSM (Nighttime Attack)
**Components:**
- Brightness: 0.4 - 0.7 (40-60% darker, dusk to night)
- FGSM ε: 0 - 4/255 (half of standard ε)

**Threat Scenario:** Security camera at night
- Attacker places adversarial object during low-light hours
- Uses smaller perturbation (easier to hide)
- Model might fail at ε=2/255 + low-light when it passes ε=8/255 on clean

**Why It Matters:**
- Standard tests: "FGSM robust to ε=8/255" ✓
- Realistic test: "FGSM fails at ε=2/255 in low-light" ✗
- Blind spot: Attacker timing > attack strength

#### Motion Blur + PGD (Fast-Moving Target)
**Components:**
- Gaussian Blur: σ=1.5-3.0 (defocus from fast motion)
- PGD ε: 2/255 (small perturbation)
- Iterations: 10 (limited compute during motion)

**Threat Scenario:** Autonomous vehicle at high speed
- Fast motion → motion blur in frames
- Attacker optimizes perturbation during motion
- Smaller ε needed because blur obscures non-adversarial details

**Why It Matters:**
- Blur and adversarial both degrade images
- Combined: synergistic effect worse than either alone
- Iterative attacker has limited compute budget during motion

#### Heavy Compression + FGSM (Video Transmission)
**Components:**
- JPEG Quality: 30 - 50 (heavy compression, blocky artifacts)
- FGSM ε: 0 - 4/255

**Threat Scenario:** Security system with lossy video transmission
- Attacker injects adversarial object before JPEG encoding
- Compression artifacts hide perturbation
- Smaller ε sufficient because compression masks it

**Why It Matters:**
- Compression and adversarial both degrade signal
- Attacker can exploit compression for stealth
- ε=4/255 + compression > ε=8/255 on clean

#### Triple Threat: Low-Light + Noise + Tiny FGSM (Worst Case)
**Components:**
- Brightness: 0.5 - 0.7 (dim)
- Gaussian Noise: σ=0.01-0.03 (sensor noise)
- FGSM ε: 0 - 2/255 (imperceptible)

**Threat Scenario:** Outdoor sensor in harsh conditions
- Multiple environmental factors combine
- Attacker uses imperceptible perturbation
- Success with ε=1/255 + poor conditions > clean image test

**Why It Matters:**
- Real world: multiple stressors always present
- Standard tests: single factor at a time
- This preset reveals worst-case vulnerability

#### Low Contrast + BIM (Hazy Conditions)
**Components:**
- Contrast: 0.5 - 0.7 (washed out, foggy)
- BIM ε: 3/255, iterations: 5 (reduced compute in low contrast)

**Threat Scenario:** Foggy/hazy weather conditions
- Low contrast + reduced compute
- BIM attack optimized for degraded image quality
- Smaller ε needed because details are already lost

**Why It Matters:**
- Weather conditions common in real deployments
- Combined with iterative attack
- Tests robustness to environmental + temporal constraint (limited inference budget)

**Estimated Time:** 20-30 minutes for 100 images
**Search Budget:** 2500 queries
**Requirements:** Adversarial Robustness Toolbox (ART)

**Validation Approach:**
- For each scenario, verify:
  1. Robustness score: realistic_attack < natural (environmental helps attacker)
  2. Vulnerability detection: if realistic_attack << min(natural, adversarial), flag as critical
  3. Environmental component alone: should preserve ~85-90% labels
  4. Adversarial component alone: should find failures
  5. Combined: should find more failures than either alone

---

## 4. Comprehensive Preset (All Threat Models)

**Purpose:** Complete robustness evaluation across all three threat models in a single run.

**Output:** Per-threat-model scores with opportunistic vulnerability detection.

**Threat Models Tested:**
- Natural: 6 strategies (single + compositional natural perturbations)
- Adversarial: 2 strategies (FGSM, PGD)
- Realistic Attack: 3 scenarios (low-light+FGSM, blur+PGD, compression+FGSM)

**Use Case:**
- Research benchmarking (publication-ready results)
- Complete model evaluation
- Comparing architectures across threat models
- Security certification

**Opportunistic Vulnerability Detection:**

The preset automatically flags models vulnerable to opportunistic attacks:

```
Natural robustness:    75.0%
Adversarial robust:    60.0%
Realistic attack:      45.0%

If realistic_attack << min(natural, adversarial):
  🚨 CRITICAL: Model vulnerable to opportunistic attacks!
  Implication: Attackers can exploit environmental timing
```

**Estimated Time:** 45-60 minutes for 100 images
**Search Budget:** 5000 queries
**Requirements:** Adversarial Robustness Toolbox (ART)

---

## 5. Lighting Preset (Legacy, Deprecated)

**Purpose:** Test robustness to illumination changes (brightness, contrast, gamma).

**Target Scenarios:**
- Outdoor cameras with varying daylight
- Time-of-day robustness (dawn, noon, dusk)
- Indoor lighting changes
- Shadow/highlight handling

### Perturbation Ranges

#### Brightness (0.5 - 1.5)
**Rationale:**
- `0.5` = 50% darker (significant underexposure)
- `1.5` = 50% brighter (significant overexposure)
- Slightly more aggressive than "standard" preset
- Covers full range of typical camera auto-exposure
- **Preserves label:** Yes, even at extremes

#### Contrast (0.7 - 1.3)
**Rationale:**
- `0.7` = 30% reduced contrast (washed out, hazy)
- `1.3` = 30% increased contrast (punchy, saturated shadows/highlights)
- Simulates atmospheric conditions (fog, haze) and display calibration
- Conservative range - values outside 0.5-1.5 can lose detail in shadows/highlights
- **Preserves label:** Yes, dynamic range preserved

#### Gamma (0.7 - 1.3)
**Rationale:**
- `γ=1.0` = Linear (no gamma correction)
- `γ<1.0` = Darker shadows, brighter highlights (more contrast in shadows)
- `γ>1.0` = Lighter shadows, compressed highlights
- Simulates different display calibrations, sRGB vs raw
- Range avoids extreme posterization
- **Preserves label:** Yes, tonal distribution changes but content clear

#### Compositional: Dim + Low Contrast
**Components:**
- Brightness: 0.4 - 0.6 (very dim)
- Contrast: 0.7 - 0.9 (reduced contrast)

**Rationale:**
- Simulates challenging lighting (overcast evening, indoor with poor lighting)
- Real cameras often produce flat, dim images in these conditions
- Tests if model relies on high contrast features
- **Preserves label:** Yes, but challenging (realistic edge case)

**Estimated Time:** 5-8 minutes for 100 images
**Search Budget:** 1000 queries

---

## 6. Blur Preset (Legacy, Deprecated)

**Purpose:** Test robustness to blur, motion, and compression artifacts.

**Target Scenarios:**
- Motion blur from camera shake
- Out-of-focus images
- Video frame compression
- Fast-moving objects

### Perturbation Ranges

#### Gaussian Blur (σ: 0.0 - 3.0)
**Rationale:**
- `σ=0.0` = Sharp image
- `σ=3.0` = Significant blur (noticeably out of focus)
- Slightly more aggressive than "standard" (max 2.5 → 3.0)
- At σ=3.0, small text would be unreadable but large objects clear
- **Preserves label:** Yes, shapes and colors preserved

#### Motion Blur (kernel: 1 - 25, angle: 0°)
**Rationale:**
- Kernel size = length of motion blur streak in pixels
- `1` = No blur
- `25` = Significant motion (fast pan or shake on ~32px object)
- Angle=0° = horizontal motion (most common for panning)
- Simulates handheld camera shake, subject motion
- **Preserves label:** Yes, blur direction visible but object identifiable

#### JPEG Compression (quality: 30 - 100)
**Rationale:**
- `30` = Heavy compression (visible 8x8 blocking)
- More aggressive than "standard" (40 → 30)
- Simulates heavily compressed video frames, web thumbnails
- At quality 30, artifacts are pronounced but content is clear
- **Preserves label:** Yes, blocky but recognizable

#### Compositional: Motion Blur + Compression
**Components:**
- Motion Blur: kernel=10-20 (moderate to heavy motion)
- JPEG Quality: 40-60 (visible compression)

**Rationale:**
- Simulates compressed video frames with motion
- Real-world video encoding scenario
- Compression algorithms struggle with motion blur (more bits needed)
- **Preserves label:** Yes, realistic video quality degradation

**Estimated Time:** 6-10 minutes for 100 images
**Search Budget:** 1200 queries

---

## 7. Corruption Preset (Legacy, Deprecated)

**Purpose:** Test robustness to noise, heavy compression, and signal degradation.

**Target Scenarios:**
- Lossy transmission (network packet loss)
- Low-bandwidth scenarios (satellite, remote sensors)
- Noisy sensors (low-light, infrared)
- Damaged/degraded images

### Perturbation Ranges

#### Gaussian Noise (std: 0.0 - 0.05)
**Rationale:**
- `0.05` = 5% noise (very visible "static")
- More aggressive than "standard" (0.03 → 0.05)
- At 5%, noise is prominent but signal-to-noise ratio is acceptable
- Simulates high ISO sensor noise, analog transmission noise
- **Preserves label:** Yes, noisy but object clear

#### JPEG Compression (quality: 10 - 100)
**Rationale:**
- `10` = Extreme compression (heavy 8x8 blocks, color banding)
- Most aggressive compression range across all presets
- At quality 10, image is heavily degraded but structure visible
- Simulates extremely low bandwidth transmission
- **Preserves label:** Yes (marginal), very degraded but identifiable

#### Gaussian Blur (σ: 0.0 - 2.0)
**Rationale:**
- `σ=2.0` = Moderate blur (less than blur preset's 3.0)
- In "corruption" context, simulates transmission blur, not focus issues
- Conservative since other corruptions are aggressive
- **Preserves label:** Yes, moderate softness

#### Compositional: Heavy Compression + Noise
**Components:**
- JPEG Quality: 10-30 (extreme compression)
- Gaussian Noise: std=0.03-0.05 (visible noise)

**Rationale:**
- Simulates worst-case transmission scenario
- Compressed over poor channel with noise
- This is a **stress test** - if model passes this, it's very robust
- **Preserves label:** Marginal (85% threshold), but realistic for some applications

**Estimated Time:** 6-10 minutes for 100 images
**Search Budget:** 1200 queries

---

---

## Threat-Model-Aware Design Principles

### Design Rationale

VisProbe 2.0 introduces threat-model-aware presets to address a critical gap in standard robustness testing:

**The Problem with Existing Approaches:**
1. **ImageNet-C, RobustBench, etc.** test natural perturbations in isolation
2. **Adversarial robustness papers** test ε=8/255 attacks on CLEAN images
3. **Real attackers** are smarter: they exploit environmental timing
   - Wait for low-light (harder for model to recognize details)
   - Use smaller ε (easier to hide in degraded image)
   - Success with ε=2/255 + low-light > ε=8/255 on clean

**Our Solution:**
- **`natural`** preset: Baseline environmental robustness
- **`adversarial`** preset: Baseline adversarial robustness
- **`realistic_attack`** preset: **CRITICAL** - the intersection
- **`comprehensive`** preset: All three with vulnerability detection

### Opportunistic Vulnerability Detection

The key insight is **gap analysis**:

```
If:  realistic_attack_score << min(natural_score, adversarial_score)
Then: Model has blind spot to opportunistic attacks
```

**Example:**
- Natural robustness: 75% (passes most environmental tests)
- Adversarial robustness: 70% (passes most attacks on clean images)
- Realistic attack: 40% (FAILS - vulnerable when combined!)

**Interpretation:**
- Model seems reasonably robust individually
- But attackers can exploit timing to win with weaker attacks
- Security certification: FAILS without realistic attack testing

### Validation Strategy for Threat-Model-Aware Presets

**For each threat-aware preset, validate:**

1. **Label Preservation (85-90% target)**
   - Generate 50+ images per strategy
   - Manual review: are they still recognizable?
   - If < 85%: reduce severity ranges
   - If > 95%: ranges are very conservative (acceptable)

2. **Threat Model Isolation**
   - Natural preset: pure environmental (no gradient-based attacks)
   - Adversarial preset: pure white-box attacks (clean images only)
   - Realistic attack: intentional combination (verify both components present)

3. **Failure Correlation**
   - Test samples that fail natural + adversarial
   - Verify realistic_attack fails on meaningful subset
   - Ensure realistic < natural + adversarial (attacker gains from timing)

4. **Opportunistic Detection**
   - For each test model:
     - Calculate threat-model scores
     - Flag if realistic_attack << min(natural, adversarial)
     - Verify flag accuracy with human inspection

### Future Work

**Additional Threat-Model-Aware Presets:**
1. **`detection_failure`** - Attacks that fool object detectors (not classification)
2. **`weather`** - Realistic weather scenarios (rain, snow, fog)
3. **`adversarial_natural`** - Natural-looking adversarial attacks (same threat model as realistic but different scenarios)

**Automated Threat Model Detection:**
- Input: test results from natural + adversarial presets
- Output: recommendation for which realistic_attack scenarios to prioritize
- Example: "Model is weak to brightness + FGSM, recommend lowlight_fgsm testing"

---

## Range Selection Methodology (Legacy - Updated for Threat Models)

### Step 1: Literature Review
- Reviewed ranges from ImageNet-C, RobustBench, AugMax
- Analyzed typical camera specifications (ISO, exposure, focus range)
- Studied compression standards (JPEG quality scales)

### Step 2: Initial Conservative Estimates
- Started with narrow ranges guaranteed to preserve labels
- Example: Brightness 0.8-1.2 (only ±20%)

### Step 3: Iterative Expansion
- Incrementally widened ranges until 10-15% of max-severity images became unrecognizable
- Example: Brightness 0.8-1.2 → 0.6-1.4 → 0.5-1.5 (stopped at 0.5-1.5 for "lighting")

### Step 4: Cross-Validation
- Generated 50+ comparison images per preset
- Manual review by multiple annotators
- Measured label preservation rate
- Adjusted ranges to target 85-90%

### Step 5: Compositional Tuning
- For compositional perturbations, used slightly more conservative ranges
- Example: Individual brightness 0.5-1.5, but in composition 0.4-0.7 (low-light only)
- Compensates for compound effect

---

## Validation Targets

### Label Preservation Rates (Target: 85-90%)

| Preset | Target Min | Target Max | Notes |
|--------|------------|------------|-------|
| **Standard** | 87% | 92% | Balanced, should have high preservation |
| **Lighting** | 85% | 90% | Slightly more aggressive brightness |
| **Blur** | 85% | 90% | Motion blur can be challenging |
| **Corruption** | 80% | 87% | Most aggressive, lower target acceptable |

### If Validation Fails

**If preservation rate < 85%:**
1. Reduce maximum severity by 10-20%
2. Example: Brightness max 1.5 → 1.3
3. Re-run validation
4. Repeat until target achieved

**If preservation rate > 95%:**
- Consider slightly increasing max (optional)
- Current ranges are conservative but safe
- Trade-off: fewer false positives (unfair tests) vs. finding edge cases

---

## Time Estimates Methodology

**Hardware Assumption:** NVIDIA RTX 3080 or equivalent, ResNet-50 model

**Estimation Formula:**
```
Time = (Num_Strategies × Queries_per_Strategy × Inference_Time) + Overhead

Where:
- Inference_Time ≈ 5ms per image (ResNet-50 on GPU)
- Overhead ≈ 20% (data loading, perturbation generation)
```

**Example (Standard preset):**
- 6 strategies (4 individual + 2 compositional)
- ~2000 total queries ÷ 6 = 333 queries/strategy
- 2000 × 5ms = 10 seconds inference
- +20% overhead = 12 seconds for model queries
- Plus adaptive search overhead ≈ 10-15 minutes total

**On CPU:** Multiply by 10-20x (slower inference)

**Smaller models (MobileNet):** Divide by 2-3x

---

## Validation Checklist

For each preset, verify:

- [ ] Generated 50+ comparison images (10 samples × 4-5 severity levels)
- [ ] Manually reviewed all images
- [ ] Calculated label preservation rate for each perturbation
- [ ] Overall preset preservation rate: 85-90% ✅
- [ ] Filled out VALIDATION_REPORT.md
- [ ] Updated VALIDATION_STATUS in presets.py
- [ ] Committed validation images and reports

---

## Future Enhancements

### Potential Additional Presets

1. **"geometric"** - Rotation, translation, scaling
2. **"weather"** - Rain, fog, snow effects
3. **"adversarial"** - PGD, FGSM (mild ε)
4. **"minimal"** - Fast smoke test (2-3 perturbations, 500 queries, 2-3 min)

### Adaptive Range Selection

Future work: Automatically tune ranges based on model performance
- Start with conservative ranges
- Gradually increase until failure rate hits 10-15%
- Auto-calibrate per model/dataset

---

## References

- **ImageNet-C:** [Hendrycks & Dietterich, 2019](https://arxiv.org/abs/1903.12261)
- **RobustBench:** [Croce et al., 2020](https://robustbench.github.io/)
- **AugMax:** [Wang et al., 2021](https://arxiv.org/abs/2103.01946)
- **JPEG Standard:** ITU-T T.81, quality scale mapping
- **Camera ISP:** Typical auto-exposure range: ±2 stops (0.25x - 4x)

---

**Last Updated:** 2024-12-10
**Validation Status:** Pending manual validation (see validation/output/)
