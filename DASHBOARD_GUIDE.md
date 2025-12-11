# VisProbe Dashboard - Feature Guide

## Overview

The VisProbe dashboard has been redesigned to provide **actionable insights** rather than just metrics. It's organized into 5 main sections that answer practitioners' key questions about model robustness.

---

## 📊 Section 1: Executive Summary

**Question Answered:** *"Is my model robust enough?"*

### What You See:
- **Robustness Score** with color-coded interpretation (Green ✅ / Yellow ⚠️ / Red ❌)
- **Critical Failures** count - how many samples have >50% confidence drop
- **Most Vulnerable Class** - which class needs the most attention

### How to Interpret:
- **Green (>80% accuracy/threshold <0.001):** Ready for deployment
- **Yellow (60-80% accuracy/threshold 0.001-0.01):** Review and consider improvements
- **Red (<60% accuracy/threshold >0.01):** Requires significant mitigation

### Action:
If your score is red/yellow, proceed directly to the "Recommended Actions" section (Section 5).

---

## 🔴 Section 2: Failure Triage

**Question Answered:** *"Which failures should I focus on first?"*

### Three Triage Strategies:

#### Tab 1: By Severity
- **Ranked list** of failures sorted by confidence drop (worst first)
- **Interactive slider** to filter by minimum confidence drop
- **Top 3 critical cases** shown with explanations and suggested fixes

**Use this to:** Find the worst cases and understand why they fail

#### Tab 2: By Class
- **Class failure summary** showing which classes are most vulnerable
- **Average confidence drop per class**
- **Drill-down** into the most vulnerable class

**Use this to:** Identify class imbalances in robustness

#### Tab 3: By Pattern
- **Metric breakdown:** Critical (>70%), High (50-70%), Medium (<50%)
- **Expandable groups** of similar failures
- **Pattern summary** for quick scanning

**Use this to:** See the distribution and patterns of failures

---

## 🔍 Section 3: Root Cause Analysis

**Question Answered:** *"Why is my model failing?"*

### Metrics Shown:
- **Pass/Fail Statistics:** Overall success rate
- **Confidence Analysis:** Average and maximum confidence drop
- **Confidence Drop Distribution:** Histogram showing how confidence degrades

### Color-Coded Insights:
- 🟢 **Green:** Low degradation (<50% avg drop) - good robustness
- 🟡 **Yellow:** Moderate degradation (50-70% drop) - needs improvement
- 🔴 **Red:** High degradation (>70% drop) - critical vulnerability

**Use this to:** Understand the severity of degradation across all samples

---

## 🎯 Section 4: Adaptive Search Analysis

**Question Answered:** *"How efficient is the adaptive search?"*

*(Only shown if search path data is available)*

### Efficiency Metrics:
- **Search Steps:** Number of iterations used
- **Grid Search Est.:** Estimated steps for equivalent grid search
- **Efficiency Gain:** How many times faster than grid search

### Visualization:
- **Convergence Plot:** Shows pass/fail points and the discovered threshold
- **Confidence Trajectory:** How model confidence changes during search

**Key Insight:** The efficiency gain demonstrates why adaptive search is better than grid search. A 50× speedup means adaptive search is dramatically more efficient.

---

## 🔧 Section 5: Recommended Actions

**Question Answered:** *"What should I do to improve robustness?"*

### Features:
- **Prioritized recommendations** (High → Medium → Low)
- **Evidence** showing why this recommendation applies
- **Expected impact** describing what improvement you can expect
- **Ready-to-use code** to implement the recommendation

### Types of Recommendations:

#### High Priority (🔴)
- **Overall robustness is low** (failure rate >50%)
- **Severe confidence degradation** (>70% drop for multiple samples)

**Action:** These require immediate attention before deployment.

#### Medium Priority (⚠️)
- **Class imbalance** (one class has >40% failure rate)
- **Moderate degradation** that can be addressed with targeted fixes

**Action:** Address these to improve overall model quality.

### How to Use:
1. Read the description to understand the problem
2. Review the evidence to verify it applies to your model
3. Copy the code example (adjust as needed for your setup)
4. Implement and retrain your model
5. Re-run the test dashboard to see improvements

---

## 📋 Additional Sections

### Metrics & Strategy
- **Key Metrics:** Technical details about the test
- **Applied Strategies:** Exact perturbations and parameters used

### Visual Comparison
- **Original Image:** What the model correctly classifies
- **Perturbed Image:** What the model struggles with
- **Residual:** Visualization of the perturbation magnitude

### Detailed Analysis
- **Search Path:** Step-by-step convergence to failure threshold
- **Ensemble Analysis:** Which layers are most affected
- **Top-K Overlap:** Prediction stability across different severity levels
- **Resolution Impact:** How robustness changes with image size
- **Noise Sensitivity:** How noise magnitude affects accuracy
- **CIFAR-10-C Corruptions:** Real-world robustness benchmark

---

## 💡 Best Practices

### 1. Start from the Top
Always begin with the Executive Summary to understand overall robustness at a glance.

### 2. Drill Down Strategically
- **If failures exist:** Use Failure Triage to identify priority cases
- **If failures are uniform:** Use Root Cause Analysis to understand degradation
- **If you have no failures:** Congratulations! Your model is robust

### 3. Action-Oriented Workflow
1. Identify problematic areas → Section 2 (Triage)
2. Understand why → Section 3 (Root Cause)
3. Get actionable fixes → Section 5 (Recommendations)
4. Implement and iterate

### 4. Class-Focused Improvement
If one class is vulnerable (Section 2, Tab 2):
1. Focus training efforts on that class
2. Apply stronger augmentation to that class
3. Collect more examples of that class
4. Re-test to verify improvement

### 5. Severity-First Prioritization
If failures are mixed (Section 3):
1. Fix critical failures first (>70% confidence drop)
2. Then address high failures (50-70% drop)
3. Finally, address medium failures (<50% drop)

---

## 📊 Understanding the Metrics

### Confidence Drop
- **Definition:** How much the model's confidence decreases after perturbation
- **Low drop (<30%):** Model remains confident, minimal robustness loss
- **Medium drop (30-70%):** Model becomes uncertain, moderate robustness loss
- **High drop (>70%):** Model is highly uncertain or reverses, severe robustness loss

### Robust Accuracy
- **Definition:** Fraction of samples that maintain correct prediction under perturbation
- **>80%:** Strong robustness
- **60-80%:** Moderate robustness
- **<60%:** Weak robustness

### Failure Threshold (ε)
- **Definition:** Minimum perturbation strength causing property failure
- **<0.001:** Very robust (high threshold)
- **0.001-0.01:** Moderately robust
- **>0.01:** Low robustness (low threshold)

---

## 🎯 Common Scenarios

### Scenario 1: "One Class Fails Frequently"
1. Go to **Failure Triage → By Class**
2. Look at the class-specific failure rate
3. Implement the "Improve [Class] Robustness" recommendation
4. Add more training samples of that class with augmentation

### Scenario 2: "Everything Fails, But Inconsistently"
1. Go to **Root Cause Analysis**
2. Check the confidence drop distribution
3. If skewed to high values: implement comprehensive augmentation (Section 5)
4. If distributed: consider ensemble methods or uncertainty calibration

### Scenario 3: "A Few Samples Fail Catastrophically"
1. Go to **Failure Triage → By Severity**
2. Look at the top critical cases
3. Implement the "Address Severe Confidence Degradation" recommendation
4. Consider confidence calibration or ensemble methods

### Scenario 4: "Model is Robust!"
1. Check **Executive Summary** - if all green, you're done!
2. Consider running with more aggressive perturbations
3. Test on additional perturbation types
4. Conduct adversarial attack evaluations

---

## 🚀 Interpreting Your Results

### Executive Summary Interpretation:

**"Robust Accuracy: 92% ✅ Strong"**
→ Your model maintains correct predictions 92% of the time under perturbation. This is production-ready.

**"Critical Failures: 3 ⚠️ Review"**
→ 3 samples have >50% confidence drop. These need attention but are not catastrophic.

**"Most Vulnerable Class: cat (5 failures)"**
→ The "cat" class is your priority. Consider adding more cat training data.

---

## 📈 Iterative Improvement Workflow

```
1. Run test dashboard
   ↓
2. Review Executive Summary
   ↓
3. Identify primary issues (via Triage & Root Cause)
   ↓
4. Implement top recommendations (via Recommendations section)
   ↓
5. Retrain model
   ↓
6. Run test dashboard again
   ↓
7. Compare improvements
   ↓
8. Repeat until satisfied with robustness
```

---

## 🤔 Frequently Asked Questions

### Q: Why should I use the severity-ranked triage instead of just looking at all failures?
**A:** Because you have limited time/resources. The severity ranking ensures you fix the worst cases first, maximizing impact per effort.

### Q: What if recommendations don't apply to my model?
**A:** The recommendations are heuristic-based. If they don't fit your scenario, you can:
1. Look at the "Evidence" section to understand the issue
2. Design a custom solution based on your domain knowledge
3. Share your findings with the VisProbe community

### Q: How often should I rerun the test?
**A:** After any:
- Model architecture change
- Training data modification
- Hyperparameter adjustment
- Augmentation strategy change

### Q: Can I use this for other perturbations?
**A:** Yes! Change the `@search` decorator parameters to test different perturbations (blur, brightness, adversarial attacks, etc.). The dashboard will automatically adapt.

---

## 📚 Additional Resources

- **VisProbe Documentation:** See project README for core concepts
- **Property-Based Testing:** Understand what you're testing with `@given` decorators
- **Adaptive Search:** Learn how binary search finds minimal failure thresholds
- **Robustness Literature:** Papers on adversarial training and certification

---

## 🎓 Learning the Dashboard

Start with:
1. **Executive Summary** (30 seconds) - understand overall status
2. **Failure Triage** (2-3 minutes) - find what to fix
3. **Root Cause Analysis** (2 minutes) - understand why
4. **Actionable Recommendations** (5 minutes) - get solutions

Total time: ~10 minutes per test to actionable insights!
