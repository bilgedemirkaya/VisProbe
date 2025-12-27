# VisProbe Documentation

**VisProbe** is a property-based robustness testing framework for vision models. Test your model against natural perturbations, adversarial attacks, and realistic attack scenarios—in just 3 lines of code.

The framework distinguishes between three **threat models** to enable realistic security testing:
- **Natural** (passive): Environmental perturbations without adversary
- **Adversarial** (active): Gradient-based attacks on clean images
- **Realistic Attack** (active + environmental): Attacks exploiting suboptimal conditions ⭐ **What standard tests miss!**

## Overview

VisProbe enables you to:

- **Test robustness across three threat models** with threat-model-aware presets
- **Detect opportunistic vulnerabilities** where attackers exploit environmental timing
- **Find failure thresholds automatically** using adaptive search
- **Export failures** for targeted retraining and analysis
- **Compare security posture** across threat models with a single function
- **Integrate into CI/CD** for continuous robustness monitoring

## Quick Example

```python
import torchvision.models as models
from visprobe import quick_check

# Load your model
model = models.resnet18(weights='DEFAULT')

# Test NATURAL robustness (environmental perturbations)
report = quick_check(model, test_data, preset="natural")
print(f"Natural robustness: {report.score:.1%}")

# View results with threat model breakdown
report.show()
```

## Key Innovation: Realistic Attack Testing

Standard robustness tests check natural and adversarial separately. **Real attackers are smarter:**

```
Standard Testing:
  ✓ FGSM ε=8/255 on clean images       (passes)
  ✓ Low-light images (passes)

Realistic Attack Testing:
  ✗ FGSM ε=2/255 on LOW-LIGHT images   (FAILS!)

Why? Attacker waits for low-light, uses SMALLER perturbation (4x easier)
```

This is what VisProbe's `realistic_attack` preset tests—and standard tools miss!

---

## Key Features

### 1. Threat-Model-Aware Presets

Choose from 4 threat-aware presets designed for different security scenarios:

#### 🌍 Natural Preset (Passive Threat Model)
Tests robustness to realistic environmental conditions without adversary.
- **Perturbations:** Brightness, contrast, blur, noise, compression, compositional
- **Use Case:** Deployment robustness, production monitoring
- **Time:** ~12-15 min for 100 images
- **Budget:** 2000 queries

```python
report = quick_check(model, data, preset="natural")
```

#### 🔐 Adversarial Preset (Active Threat Model)
Tests robustness to white-box gradient-based attacks on clean images.
- **Attacks:** FGSM, PGD, BIM with standard ε=8/255
- **Use Case:** Security testing, adversarial ML research
- **Time:** ~15-25 min for 100 images
- **Budget:** 1500 queries
- **Requires:** `pip install adversarial-robustness-toolbox`

```python
report = quick_check(model, data, preset="adversarial")
```

#### ⭐ Realistic Attack Preset (Active + Environmental)
**THE CRITICAL PRESET.** Tests adversarial attacks under suboptimal environmental conditions.
- **Scenarios:** Low-light + FGSM, blur + PGD, compression + FGSM, etc.
- **Use Case:** Security-critical deployments, autonomous vehicles, surveillance
- **Time:** ~20-30 min for 100 images
- **Budget:** 2500 queries
- **Requires:** ART

```python
report = quick_check(model, data, preset="realistic_attack")

# Check for opportunistic vulnerability
if report.vulnerability_warning:
    print("🚨 CRITICAL:", report.vulnerability_warning)
```

#### 📊 Comprehensive Preset (All Threat Models)
Complete evaluation across all three threat models with per-threat-model breakdown.
- **Output:** Separate scores for natural, adversarial, realistic_attack
- **Use Case:** Research, benchmarking, publication-ready results
- **Time:** ~45-60 min for 100 images
- **Budget:** 5000 queries

```python
report = quick_check(model, data, preset="comprehensive")

# Per-threat-model scores
print(report.threat_model_scores)
# Output: {'natural': 0.75, 'adversarial': 0.60, 'realistic_attack': 0.45}

# Automatic vulnerability detection
if report.vulnerability_warning:
    print("Model vulnerable to opportunistic attacks!")
```

### 2. Opportunistic Vulnerability Detection

Automatically detects when models are vulnerable to timing-based attacks:

```python
# Test all three threat models
results = compare_threat_models(model, data, budget=1000)

# Output: Comparison showing gap analysis
# Natural:        75%
# Adversarial:    70%
# Realistic:      40%  ← Much lower! Vulnerable to opportunistic attacks
```

**What it means:**
- Model seems reasonably robust individually
- But attackers can exploit environmental conditions to win with smaller perturbations
- Security assessment: **INCOMPLETE** without realistic attack testing

### 3. Compositional Perturbations

Uniquely tests multiple perturbations together to catch real-world failure modes:
- Low-light + blur (nighttime handheld photos)
- Compression + noise (low-bandwidth transmission)
- Low-contrast + adversarial (foggy/hazy conditions)
- Triple threat: low-light + noise + tiny FGSM (worst case)

### 4. Adaptive Search

Efficiently finds exact failure thresholds for each perturbation:
- Binary search-like algorithm
- Adaptive step-size refinement
- Per-sample failure tracking
- Handles both natural and adversarial perturbations

### 5. Actionable Results

```python
# Score breakdown
print(f"Overall: {report.score:.1%}")
print(f"Threat model scores: {report.threat_model_scores}")

# Failure analysis
for failure in report.failures[:5]:
    print(f"Sample {failure['index']}: {failure['original_pred']} → {failure['perturbed_pred']}")

# Export for retraining
report.export_failures(n=20, output_dir="./hard_cases")

# Get summary dict
summary = report.summary
# {'score': 0.60, 'threat_model': 'all', 'threat_model_scores': {...}, ...}
```

---

## Installation

```bash
# From source
git clone https://github.com/bilgedemirkaya/VisProbe.git
cd VisProbe
pip install -e .

# For adversarial testing (required for adversarial/realistic_attack presets)
pip install adversarial-robustness-toolbox
```

---

## Documentation Structure

- **[User Guide](user-guide.md)** - Getting started and common workflows
- **[Threat-Model Design](../PRESET_DESIGN.md)** - Detailed technical design of threat-aware presets (root docs)
- **[API Reference](../COMPREHENSIVE_API_REFERENCE.md)** - Detailed API documentation (root docs)
- **[Troubleshooting](../TROUBLESHOOTING.md)** - Common issues and solutions (root docs)
- **[Main README](../README.md)** - Project overview and quick start (root docs)

---

## Use Cases

### 1. Testing All Threat Models

```python
from visprobe import quick_check

# 1. Natural robustness (deployment baseline)
natural = quick_check(model, data, preset="natural")

# 2. Adversarial robustness (security hardening)
adversarial = quick_check(model, data, preset="adversarial")

# 3. Realistic attack (the critical one!)
realistic = quick_check(model, data, preset="realistic_attack")

# Check for blind spot
if realistic.score < min(natural.score, adversarial.score) - 0.1:
    print("🚨 Model vulnerable to opportunistic attacks!")
```

### 2. Production Deployment Validation

```python
def validate_deployment():
    model = load_production_model()
    test_data = load_test_data()

    # Quick natural robustness check
    report = quick_check(model, test_data, preset="natural", budget=1000)

    # Enforce robustness requirement
    assert report.score > 0.75, f"Model too fragile: {report.score:.1%}"

    return report
```

### 3. Security-Critical System Testing

```python
# For autonomous vehicles, surveillance, etc.
# Test realistic attack scenarios where model might fail
report = quick_check(
    model,
    data,
    preset="realistic_attack",
    budget=2500
)

# Check for vulnerabilities
if report.vulnerability_warning:
    raise SecurityError("Model has critical blind spot to opportunistic attacks")
```

### 4. Model Comparison Across Threat Models

```python
from visprobe import compare_threat_models

results = compare_threat_models(model_v1, data, budget=1000)
results2 = compare_threat_models(model_v2, data, budget=1000)

print("Model V1:")
for tm, score in results['scores'].items():
    print(f"  {tm}: {score:.1%}")

print("Model V2:")
for tm, score in results2['scores'].items():
    print(f"  {tm}: {score:.1%}")

# Model v2 is more vulnerable to realistic attacks
if results2['scores']['realistic_attack'] < results['scores']['realistic_attack']:
    print("V2 has worse opportunistic attack resilience")
```

### 5. Targeted Retraining

```python
# Find worst cases for each threat model
report = quick_check(model, data, preset="comprehensive")

if report.threat_model_scores['realistic_attack'] < 0.50:
    # Export realistic attack failures for retraining
    failures_path = report.export_failures(n=100)
    print(f"Add {failures_path} to training set to improve opportunistic attack resistance")
```

### 6. CI/CD Integration

```yaml
# In your CI pipeline (e.g., GitHub Actions)
- name: Test robustness
  run: |
    python -c "
    from visprobe import quick_check
    import your_model

    report = quick_check(
        your_model.load(),
        your_model.get_test_data(),
        preset='comprehensive'
    )

    # Enforce multi-threat-model requirements
    assert report.threat_model_scores['natural'] > 0.70
    assert report.threat_model_scores['adversarial'] > 0.60
    assert report.threat_model_scores['realistic_attack'] > 0.50

    print('✅ All robustness checks passed')
    "
```

---

## Backward Compatibility

The original presets (`standard`, `lighting`, `blur`, `corruption`) are still available but deprecated:

```python
report = quick_check(model, data, preset="standard")
# DeprecationWarning: Consider using 'natural' for similar results
# or 'comprehensive' for complete testing
```

**Migration:**
- Use `natural` instead of `standard`, `lighting`, `blur`, `corruption`
- Use `comprehensive` for complete threat-model-aware evaluation

---

## Quick Links

- **[Main README](../README.md)** - Project overview and quick start
- **[Threat-Model Design](../PRESET_DESIGN.md)** - Why realistic_attack matters
- **[GitHub Repository](https://github.com/bilgedemirkaya/VisProbe)** - Source code and examples
- **[Report Issues](https://github.com/bilgedemirkaya/VisProbe/issues)** - Bug reports

---

## Key Insight: The Opportunistic Attack Problem

Standard robustness testing has a critical blind spot:

**What existing tools test:**
```
✓ Can your model handle low-light? (environmental test)
✓ Can your model resist FGSM ε=8/255? (adversarial test)
```

**What they miss:**
```
✗ Can your model resist FGSM ε=2/255 in low-light? (opportunistic attack)
```

**Why it matters:**
- Attackers observe environmental conditions
- They use SMALLER perturbations when conditions help them
- Success with 4x smaller ε + environmental + timing > larger ε on clean images

**VisProbe's solution:**
- `natural` preset: environmental baseline
- `adversarial` preset: adversarial baseline
- `realistic_attack` preset: the intersection (what matters!)
- Automatic vulnerability detection: gap analysis between them

---

## Citation

If you use VisProbe in your research, please cite:

```bibtex
@software{visprobe2025,
  title={VisProbe: Threat-Model-Aware Robustness Testing for Vision Models},
  author={Bilge Demirkaya},
  year={2025},
  url={https://github.com/bilgedemirkaya/VisProbe}
}
```

---

## Next Steps

- **New to VisProbe?** Start with the [User Guide](user-guide.md)
- **Want to understand threat models?** Read [PRESET_DESIGN.md](../PRESET_DESIGN.md)
- **Have your own model?** Check the [examples/ folder](https://github.com/bilgedemirkaya/VisProbe/tree/main/examples) on GitHub
- **Need to test all threat models?** Use the `compare_threat_models()` function (see User Guide)
- **Still have questions?** Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

---

**Latest Update:** 2025-12-26 - Updated for threat-model-aware preset system (VisProbe 2.0)
