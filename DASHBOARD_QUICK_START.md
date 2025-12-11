# Dashboard Quick Start - 10-Minute Guide

## What You'll See (New Structure)

When you run the dashboard, you'll see reports organized into 5 actionable sections:

---

## ⚡ TL;DR - Read This First

1. **Open the dashboard** → Look at Executive Summary (30 seconds)
2. **If all green?** → You're done, model is robust! 🎉
3. **If yellow/red?** → Go to "Recommended Actions" section (5 minutes)
4. **Copy the code example** → Implement the suggestion (copy-paste ready)
5. **Retrain your model** → Re-run test to verify improvement

---

## 📖 The 5 Sections Explained

### Section 1️⃣: Executive Summary (30 seconds)

**What you see:**
```
📊 Executive Summary

🔒 Robustness Score: 85%
✅ Status: Strong robustness - Ready for evaluation

Critical Failures: 2
⚠️ Alert: Review these edge cases

Most Vulnerable Class: rare_bird (1 failure)
💡 Recommendation: Consider adding more training data
```

**What it means:**
- Green ✅ = Your model is robust enough
- Yellow ⚠️ = Review before deployment
- Red ❌ = Needs improvement

**Action:** If yellow or red, continue to next sections. If green, you're done!

---

### Section 2️⃣: Failures to Investigate (2 minutes)

**Three tabs to explore:**

#### Tab A: By Severity (Worst First)
```
Showing 5/100 failures

Rank | Original | Predicted | Conf. Drop | Severity
-----|----------|-----------|------------|----------
1    | dog      | cat       | 95%        | 🔴 Critical
2    | dog      | cat       | 87%        | 🔴 Critical
3    | bird     | plane     | 62%        | 🟠 High
4    | cat      | dog       | 45%        | 🟡 Medium
5    | bird     | cat       | 32%        | 🟡 Medium
```

**Action:** Focus on cases with >70% drop first (critical failures)

#### Tab B: By Class (Find Weak Classes)
```
Class | Failures | Avg. Drop | Severity
------|----------|-----------|----------
dog   | 3        | 78%       | 🔴 Critical
bird  | 2        | 47%       | 🟡 Medium
cat   | 0        | —         | ✅ Good

💡 "dog" is your priority. Consider adding more training data.
```

**Action:** Identify the worst-performing class and focus on it

#### Tab C: By Pattern (See the Distribution)
```
🔴 Critical Failures (>70% drop): 2 cases
🟠 High Priority (50-70% drop): 1 case
🟡 Medium Priority (<50% drop): 2 cases
```

**Action:** Understand the distribution (all critical = urgent, all medium = less urgent)

---

### Section 3️⃣: Root Cause Analysis (1 minute)

**What you see:**
```
Pass/Fail Statistics:
├── Samples Passed: 95/100 (95%)
└── Samples Failed: 5/100 (5%)

Confidence Analysis:
├── Avg Confidence Drop: 45%
└── Max Confidence Drop: 95%

[Histogram showing distribution of confidence drops]

✅ Insight: "Low Degradation - Average 45% drop indicates good robustness"
```

**What it means:**
- If average drop < 50% → Good robustness ✅
- If average drop 50-70% → Moderate robustness ⚠️
- If average drop > 70% → Poor robustness ❌

**Action:** Understand the overall severity of failures

---

### Section 4️⃣: Adaptive Search Analysis (Optional)

**What you see (if search-based test):**
```
Efficiency Metrics:
├── Search Steps: 24 (iterations used)
├── Grid Search Est.: 1200 (estimated steps if using grid)
└── Efficiency Gain: 50× (much faster!)

[Visualization of convergence]

💡 Found failure threshold at ε = 0.00342 in 24 steps
   Grid search would need 1200 steps for same precision!
```

**What it means:** Shows how adaptive search is more efficient than alternatives

**Action:** Appreciate how smart the search algorithm is! (No action needed)

---

### Section 5️⃣: Recommended Actions (5 minutes)

**What you see (prioritized, with code ready):**

#### 🔴 High Priority 1: Improve "dog" Robustness

**Why?**
```
Evidence: Dog class has 3 failures (60% of all failures)
Impact: Class-specific training typically improves robustness 15-25%
```

**How to implement:**
```python
# Step 1: Export dog training samples
dog_data = [(x, y) for x, y in dataset if y == 'dog']

# Step 2: Apply stronger augmentation
strong_aug = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1))
])

# Step 3: Retrain with augmented data
model.train()
for epoch in range(5):
    for x, y in dog_data:
        x_aug = strong_aug(x)
        logits = model(x_aug)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Action:** Copy the code, adapt to your setup, retrain, re-test!

---

#### ⚠️ Medium Priority 2: Address Severe Confidence Degradation

**Why?**
```
Evidence: 2 samples show >90% confidence drop
Impact: Confidence calibration can improve prediction certainty
```

**How to implement:**
```python
# Temperature scaling for calibration
temperature = 1.5  # Adjust this parameter
calibrated_logits = logits / temperature
calibrated_probs = torch.softmax(calibrated_logits, dim=1)

# The model becomes more uncertain (better calibrated)
# High confidence → only when truly confident
```

---

## 🎯 The Complete Workflow

```
┌─ START: Run your test
│
├─ Section 1: Executive Summary
│   └─> All green? → END, you're done! 🎉
│
├─ If issues found:
│   ├─ Section 2: Failure Triage
│   │   └─> "Which failures matter most?"
│   │
│   ├─ Section 3: Root Cause Analysis
│   │   └─> "How severe is the degradation?"
│   │
│   └─ Section 5: Recommended Actions
│       └─> "Here's the fix with code"
│
└─ Implement + Retrain → Re-run test → Compare
```

---

## 🚀 Command Reference

### Run a test:
```bash
# Your test file (uses @model, @given, @search decorators)
python path/to/your_test.py
```

### View results:
```bash
# View the interactive dashboard
streamlit run src/visprobe/cli/dashboard.py -- path/to/your_test.py
```

### What gets saved:
```
results/
├── test_name.json         # Full test report
├── test_name.csv          # Per-sample metrics
└── images/                # Test images (if saved)
```

---

## 💡 Tips & Tricks

### Tip 1: Use the Sliders
In "Failures to Investigate" → "By Severity", use the confidence drop slider to focus on specific severity levels.

### Tip 2: Export Failures
To create a training set from failures, save the reports:
```python
# In your code, iterate failures from the report
failures = [s for s in report["per_sample_metrics"] if not s["passed"]]
# Add these to your training dataset
```

### Tip 3: Compare Before/After
Run the test twice (before and after retraining) and compare:
- Section 1: Robustness score improvements
- Section 2: Fewer failures overall
- Section 5: Different recommendations

### Tip 4: Test Multiple Perturbations
Modify your `@search` decorator to test different perturbations:
```python
@search(Brightness(0, 0.5))      # Brightness
@search(Blur(0, 3))               # Blur
@search(Noise(0, 0.1))            # Noise
@search(RotationStrategy(0, 30))  # Rotation
```

### Tip 5: Check Different Class Combinations
If you have multi-class data, run separate tests per class to identify specific weaknesses.

---

## ❓ Quick FAQs

**Q: How long should the dashboard take to view?**
A:
- Executive Summary: 30 seconds
- Failure Triage: 1-2 minutes
- Root Cause Analysis: 1 minute
- Recommendations: 2-3 minutes
- **Total: ~10 minutes to actionable insight**

**Q: What if I don't have failures?**
A: Congratulations! Your model is robust to that perturbation. Try:
- Stronger perturbations (higher epsilon)
- Different perturbation types
- Adversarial attacks

**Q: Can I ignore yellow/low priority items?**
A: Only if:
1. Your overall robustness (Section 1) is green, AND
2. The number of failures is small (<2%), AND
3. You don't care about edge cases

Otherwise, implement them for production readiness.

**Q: How many times should I iterate?**
A: Typically 2-3 rounds:
1. Run test → Get recommendations
2. Implement top recommendations
3. Run test again → See improvement
4. Repeat if needed

---

## 🎓 Learning Path

### First Time Using Dashboard:
1. Read **DASHBOARD_GUIDE.md** sections 1-3 (overview)
2. Run your first test
3. Follow the 5 sections top-to-bottom

### Regular Usage:
1. Check Executive Summary (30 seconds)
2. Skip to Recommended Actions (if issues found)
3. Implement suggested fixes

### Advanced Usage:
1. Use multiple perturbation types (compare dashboard results)
2. Analyze threshold distribution (Section 4)
3. Correlate class vulnerabilities with data characteristics

---

## 📞 Getting Help

If something doesn't make sense:
1. Check **DASHBOARD_GUIDE.md** for detailed explanations
2. Look at **Common Scenarios** section for your use case
3. Check the specific section tabs for more information
4. Each metric has tooltip help (hover over `?` icons)

---

## ✨ You're Ready!

```
✅ You understand the 5 sections
✅ You know what to look for
✅ You have code examples to implement
✅ You can iterate and improve

→ Go run your test and improve your model's robustness!
```

Happy testing! 🚀
