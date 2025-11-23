# VisProbe Test Documentation

This document provides information about each test file created to test in the VisProbe framework use cases, explaining their purpose, strategies, properties, and expected results.

## Overview

Each test file demonstrates different aspects of the framework:

- **Adversarial Attacks**: FGSM (Fast Gradient Sign Method)
- **Natural Perturbations**: Gaussian noise
- **Properties**: Label consistency, confidence preservation, top-k overlap, output stability
- **Adaptive Search**: Finding failure thresholds automatically

## 🧪 Test Files

### 1. `test_fgsm_strict.py` - Basic Adversarial Testing

**Purpose**: Demonstrates the most basic form of adversarial robustness testing using FGSM with a single, strict property.

**Strategy**:
- **Attack**: `FGSMStrategy(eps=0.03)` - Fast Gradient Sign Method with 3% perturbation
- **Property**: `label_constant` only - prediction must stay the same
- **Dataset**: CIFAR-10 test set (256 samples)
- **Model**: Pre-trained ResNet-56 with normalization

**Expected Results**:
- **Baseline accuracy**: ~93% (clean data)
- **Robust accuracy**: ~30-40% (under FGSM attack)
- **Behavior**: Most samples fail because of perturbation.

**Usage**:
```bash
python test_fgsm_strict.py
visprobe visualize test_fgsm_strict.py
```

**Learning Points**:
1. FGSM is effective at causing label changes even with small perturbations
2. Label consistency is a strict property - many models fail this test
3. This provides a baseline for understanding model vulnerability

---

### 2. `test_fgsm_comprehensive.py` - Multi-Property Testing

**Purpose**: Demonstrates comprehensive adversarial robustness evaluation using FGSM with multiple properties simultaneously.

**Strategy**:
- **Attack**: `FGSMStrategy(eps=0.03)` - Same FGSM attack
- **Properties**: All 4 properties tested simultaneously:
  1. `label_constant` - prediction must stay the same
  2. `confidence_drop(max_drop=0.5)` - confidence can't drop >50%
  3. `topk_overlap_rate(k=5, min_overlap=3)` - 3/5 top predictions must overlap
  4. `delta_output(max_delta=2.0)` - output vector L2 distance limit
- **Dataset**: CIFAR-10 test set (256 samples)
- **Model**: Pre-trained ResNet-56 with normalization

**Expected Results**:
- **Baseline accuracy**: ~93% (clean data)
- **Robust accuracy**: ~0-5% (much lower than single property test)
- **Behavior**: Very few samples pass all properties due to strict criteria

**Usage**:
```bash
python test_fgsm_comprehensive.py
visprobe visualize test_fgsm_comprehensive.py
```

**Learning Points**:
1. Multiple properties catch more failure modes than single properties
2. Comprehensive testing reveals deeper vulnerabilities
3. Different properties fail for different reasons (label change vs confidence drop)
4. This shows the importance of testing multiple robustness criteria

---

### 3. `test_gaussian_noise.py` - Natural Perturbation Testing

**Purpose**: Demonstrates robustness testing against natural perturbations (Gaussian noise) with more lenient properties.

**Strategy**:
- **Attack**: Custom `GaussianNoiseStrategy(std=0.05)` - Add Gaussian noise with 5% standard deviation
- **Properties**: Lenient properties for natural perturbations:
  1. `confidence_drop(max_drop=0.3)` - 30% confidence drop allowed
  2. `topk_overlap_rate(k=5, min_overlap=3)` - 3/5 top predictions must overlap
  3. `delta_output(max_delta=1.5)` - smaller output change limit
- **Dataset**: CIFAR-10 test set (256 samples)
- **Model**: Pre-trained ResNet-56 with normalization

**Expected Results**:
- **Baseline accuracy**: ~93% (clean data)
- **Robust accuracy**: ~5-15% (higher than adversarial tests)
- **Behavior**: More samples pass due to lenient criteria for natural noise

**Usage**:
```bash
python test_gaussian_noise.py
visprobe visualize test_gaussian_noise.py
```

**Learning Points**:
1. Natural perturbations are less aggressive than adversarial attacks
2. Lenient properties are appropriate for natural noise
3. Models can be more robust to natural perturbations than adversarial ones
4. This provides a baseline for "real-world" robustness

---

### 4. `test_adaptive_search.py` - Adaptive Failure Search

**Purpose**: Demonstrates adaptive search functionality to automatically find the minimum perturbation level that causes failures.

**Strategy**:
- **Attack**: Adaptive `FGSMStrategy(eps=variable)` - FGSM with varying epsilon
- **Property**: `label_constant` - prediction must stay the same
- **Search Parameters**:
  - `initial_level=0.01` - Start with 1% perturbation
  - `step=0.02` - Initial step size of 2%
  - `min_step=0.005` - Minimum step size of 0.5%
  - `max_queries=30` - Maximum 30 model queries
- **Dataset**: CIFAR-10 test set (64 samples - smaller for efficiency)
- **Model**: Pre-trained ResNet-56 with normalization

**Expected Results**:
- **Baseline accuracy**: ~92% (clean data)
- **Failure threshold**: ε ≈ 0.001-0.01 (very small perturbation)
- **Queries used**: 5-30 (efficient search)
- **Behavior**: Finds the exact perturbation level where failures begin

**Usage**:
```bash
python test_adaptive_search.py
visprobe visualize test_adaptive_search.py
```

**Learning Points**:
1. Adaptive search efficiently finds failure thresholds
2. Very small perturbations can cause failures in vulnerable models
3. This provides quantitative measures of model sensitivity
4. Search is much more efficient than brute force testing

---

## 🔧 Available Properties

VisProbe provides several robustness properties that can be combined:

### `label_constant(clean_outputs, pert_outputs)`
- **Purpose**: Check if the top-1 prediction label stays the same
- **Strictness**: Very strict - any label change is a failure
- **Use case**: Basic adversarial robustness testing

### `confidence_drop(clean_outputs, pert_outputs, max_drop=0.3)`
- **Purpose**: Ensure model confidence doesn't drop too much
- **Parameters**: `max_drop` - maximum allowed confidence drop (0.0-1.0)
- **Use case**: Testing confidence preservation under perturbations

### `topk_overlap_rate(clean_outputs, pert_outputs, k=5, min_overlap=3)`
- **Purpose**: Check if top-k predictions remain consistent
- **Parameters**:
  - `k` - number of top predictions to consider
  - `min_overlap` - minimum number that must overlap
- **Use case**: Testing prediction stability beyond just top-1

### `delta_output(clean_outputs, pert_outputs, max_delta=1.0, norm_type="l2")`
- **Purpose**: Check if output vectors remain similar
- **Parameters**:
  - `max_delta` - maximum allowed L2 distance
  - `norm_type` - norm type ("l2", "l1", "linf")
- **Use case**: Testing output stability at the vector level

### `self_ensemble(clean_outputs, pert_outputs, **kwargs)`
- **Purpose**: Advanced layer consistency analysis
- **Parameters**: Various ensemble and consistency parameters
- **Use case**: Advanced robustness analysis (not used in these examples)

---

## 🎯 Available Strategies

### Adversarial Attacks

#### `FGSMStrategy(eps=0.03)`
- **Type**: Fast Gradient Sign Method
- **Purpose**: Generate adversarial examples using gradient information
- **Parameters**: `eps` - perturbation size (0.0-1.0)
- **Use case**: Standard adversarial testing

#### `PGDStrategy(eps=0.03, eps_step=0.1, nb_iter=40)`
- **Type**: Projected Gradient Descent
- **Purpose**: Iterative adversarial attack
- **Parameters**:
  - `eps` - maximum perturbation
  - `eps_step` - step size per iteration
  - `nb_iter` - number of iterations
- **Use case**: More powerful adversarial testing

#### `DeepFoolStrategy(max_iter=100, epsilon=1e-6)`
- **Type**: DeepFool attack
- **Purpose**: Find minimal adversarial perturbation
- **Parameters**:
  - `max_iter` - maximum iterations
  - `epsilon` - convergence threshold
- **Use case**: Finding minimal adversarial examples

### Natural Perturbations

#### `GaussianNoiseStrategy(std=0.05)`
- **Type**: Custom Gaussian noise
- **Purpose**: Add random Gaussian noise to images
- **Parameters**: `std` - standard deviation of noise
- **Use case**: Testing robustness to natural noise

---

## 🚀 Running All Tests

```bash
# Run all tests
python test_fgsm_strict.py
python test_fgsm_comprehensive.py
python test_gaussian_noise.py
python test_adaptive_search.py

# Visualize all results
visprobe visualize test_fgsm_strict.py
visprobe visualize test_fgsm_comprehensive.py
visprobe visualize test_gaussian_noise.py
visprobe visualize test_adaptive_search.py
```
