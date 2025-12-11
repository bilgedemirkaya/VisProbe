# Dashboard Rewrite: From Metrics to Actionable Insights

## What Changed

The VisProbe dashboard has been completely redesigned to answer practitioners' actual questions instead of just displaying metrics.

---

## 🔄 Before vs After

### BEFORE: Metric Dumping
```
📊 Key Metrics
├── Robust Accuracy: 75%
├── Failure Threshold: 0.003
├── Model Queries: 1250
├── Runtime: 45.2s
└── (No interpretation)

📈 Detailed Analysis
├── Search Path (line chart)
├── Ensemble Analysis
├── Top-K Overlap
└── Raw JSON Report
```

**Problems:**
- Users had to interpret numbers themselves
- No guidance on what to do
- All information at once (overwhelming)
- No failure ranking or prioritization
- No actionable recommendations

---

### AFTER: 5-Section Structured Insight Flow

#### **Section 1: Executive Summary**
```
📊 Executive Summary
├── Robustness Score: 75% ⚠️ Moderate Robustness
│   └── Status: "Review findings before deployment"
├── Critical Failures: 12
│   └── Alert: "12 images show severe robustness issues"
└── Most Vulnerable Class: cat (8 failures)
    └── Recommendation: "Focus efforts on improving cat robustness"
```

**Value:** Instant understanding of overall status + color-coded action signal

#### **Section 2: Failure Triage** ← NEW
```
🔴 Failures to Investigate
├── Tab 1: By Severity
│   ├── Ranked list (worst first)
│   ├── Confidence drop filter slider
│   └── Top 3 critical cases with explanations
│
├── Tab 2: By Class
│   ├── Class failure summary table
│   ├── Average confidence drop per class
│   └── Drill-down into most vulnerable class
│
└── Tab 3: By Pattern
    ├── Critical/High/Medium breakdown
    └── Expandable failure groups
```

**Value:** Users know exactly which failures to investigate first (no guessing)

#### **Section 3: Root Cause Analysis** ← NEW
```
🔍 Root Cause Analysis
├── Pass/Fail Statistics
├── Confidence Analysis (avg + max drop)
├── Confidence Drop Distribution (histogram)
└── Color-coded Insights
    ├── 🟢 Green: "Low degradation - good robustness"
    ├── 🟡 Yellow: "Moderate degradation - needs improvement"
    └── 🔴 Red: "High degradation - critical vulnerability"
```

**Value:** Understand patterns and severity of failures systematically

#### **Section 4: Adaptive Search Analysis**
```
🎯 Adaptive Search Analysis
├── Efficiency Metrics
│   ├── Search Steps: 32
│   ├── Grid Search Est.: 1600
│   └── Efficiency Gain: 50×
├── Convergence Visualization
│   └── Shows pass/fail points and threshold discovery
└── Insight: "Search converged after 32 steps, finding failure threshold..."
```

**Value:** Demonstrates why adaptive search is valuable (not just a number)

#### **Section 5: Actionable Recommendations** ← NEW
```
🔧 Recommended Actions
├── 🔴 High: "Overall Robustness is Low (60% failure rate)"
│   ├── Evidence: "60/100 samples failed"
│   ├── Expected Impact: "Augmentation can improve robustness 10-30%"
│   └── How to Implement: [Ready-to-use code with transforms.Compose]
│
├── ⚠️ Medium: 'Improve "cat" Robustness'
│   ├── Evidence: "8 cat failures vs 2 avg per other class"
│   ├── Expected Impact: "Class-specific training improves cat robustness 15-25%"
│   └── How to Implement: [Filter training data for cats, apply augmentation]
│
└── 🔴 High: "Address Severe Confidence Degradation"
    ├── Evidence: "15 samples show >70% confidence drop"
    ├── Expected Impact: "Temperature scaling improves calibration"
    └── How to Implement: [Confidence calibration code example]
```

**Value:** Users get ranked, evidence-based recommendations with ready-to-use code

---

## 📊 Concrete Example

### Old Dashboard Output:
```
Robust Accuracy: 75%
Critical Failures: 12
Most Vulnerable Class: cat
```

**User's question:** "OK, what do I do now?" 😕

---

### New Dashboard Output:

**Section 1 (10 seconds):**
```
Robustness Score: 75% ⚠️ Moderate
Critical Failures: 12
Most Vulnerable Class: cat (8 failures)

⚠️ "Review findings before deployment"
```

**Section 2 - Severity Tab (1 minute):**
```
Showing 12/100 failures

Rank | Original | Predicted | Conf. Drop | Severity
-----|----------|-----------|------------|----------
1    | cat      | dog       | 92%        | 🔴 Critical
2    | cat      | dog       | 87%        | 🔴 Critical
3    | dog      | cat       | 78%        | 🔴 Critical
...

Top Critical Case: Cat image misclassified as dog with 92% confidence drop
Why: Model became extremely uncertain
Actions: Add more training samples, use stronger augmentation
```

**Section 2 - Class Tab (30 seconds):**
```
Class  | Failures | Avg. Drop | Severity
-------|----------|-----------|----------
cat    | 8        | 85%       | 🔴 Critical
dog    | 3        | 62%       | 🟠 High
bird   | 1        | 45%       | 🟡 Medium

Focus: "cat" class is your priority. Consider adding more training data.
```

**Section 5 (2 minutes):**
```
🔴 High Priority 1: Improve "cat" Robustness
Evidence: Cat has 8 failures, 80% failure rate
Expected Impact: Class-specific training improves robustness 15-25%

How to Implement:
```python
# Export cat training samples
cat_data = [(x, y) for x, y in dataset if y == 'cat']

# Apply stronger augmentation
strong_aug = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.GaussianBlur(kernel_size=3),
])

# Retrain with cat-specific augmentation
```
```

**User's question:** "Perfect, I know exactly what to do!" ✅

---

## 🎯 Key Improvements

### 1. **Interpretability**
- Before: "75% accuracy" - what does this mean?
- After: "75% ⚠️ Moderate - review before deployment" - crystal clear

### 2. **Prioritization**
- Before: "12 failures" - which ones matter?
- After: Ranked by severity, color-coded (Critical → High → Medium)

### 3. **Actionability**
- Before: User has to figure out what to do
- After: Top-ranked recommendations with ready-to-use code

### 4. **Efficiency**
- Before: 5+ minutes to understand findings
- After: 10 minutes to actionable insights (includes implementation time)

### 5. **Navigation**
- Before: Everything at once (overwhelming)
- After: Logical flow (summary → details → actions)

---

## 🆕 New Features

### Failure Triage (3 Views)
1. **By Severity** - worst-first ranking
2. **By Class** - which classes need help
3. **By Pattern** - distribution and grouping

### Root Cause Analysis
- Confidence drop distribution histogram
- Pass/fail statistics
- Color-coded insights

### Recommendations
- Prioritized (High → Medium → Low)
- Evidence-based (not guesses)
- Code-ready (copy-paste implementation)

### Search Path Insights
- Efficiency metrics (steps vs grid search)
- Convergence visualization
- Estimated speedup calculation

---

## 📈 Usage Statistics Projection

### Time to Insight:
- **Old Dashboard:** 5-10 minutes of scrolling + manual interpretation
- **New Dashboard:** 2-3 minutes following the structure

### Action Clarity:
- **Old Dashboard:** "What should I improve?" (requires domain expertise)
- **New Dashboard:** "Here are the top 3 things to improve" (guided path)

### Code Examples:
- **Old Dashboard:** 0 (user must write from scratch)
- **New Dashboard:** 3-5 ready-to-adapt code snippets

---

## 🔧 Technical Implementation

### New Functions in `dashboard_helpers.py`:

```python
# Executive Summary
render_executive_summary()

# Failure Analysis & Triage
extract_failures_from_report()
render_failure_triage()

# Root Cause
render_root_cause_analysis()

# Search Path
render_search_path_insights()
estimate_grid_search_equivalent()

# Recommendations
render_actionable_recommendations()
generate_recommendations()
```

### New Main Structure in `dashboard.py`:

```python
def render_all_reports_new(results: dict):
    # Section 1: Executive Summary
    render_executive_summary(report)

    # Section 2: Failure Triage
    render_failure_triage(report)

    # Section 3: Root Cause Analysis
    render_root_cause_analysis(report)

    # Section 4: Search Path (if available)
    if report.get("search_path"):
        render_search_path_insights(report)

    # Section 5: Actionable Recommendations
    render_actionable_recommendations(report)

    # Additional Details (tabbed interface)
    render_test_details()
```

---

## 📚 Documentation

A comprehensive **DASHBOARD_GUIDE.md** has been created covering:
- How to interpret each section
- Best practices for using the dashboard
- Common scenarios and how to address them
- FAQ and troubleshooting
- Learning path for new users

---

## ✨ Design Principles

The new dashboard follows these principles:

1. **Answer Real Questions**
   - Practitioners don't care about metrics
   - They care about: "Is it robust?" "What's broken?" "What do I fix?"

2. **Actionable Over Beautiful**
   - A table showing which class fails is more valuable than a pretty pie chart
   - Code snippets beat high-level suggestions

3. **Guided Navigation**
   - Users follow a logical flow (summary → details → action)
   - No decision fatigue from information overload

4. **Evidence-Based Recommendations**
   - Every suggestion shows the data supporting it
   - Users can verify/adjust recommendations for their context

5. **Progressive Disclosure**
   - Quick summary first (10 seconds)
   - Drill into details if interested (2-3 minutes)
   - Deep analysis for research (10+ minutes)

---

## 🚀 Getting Started

To use the new dashboard:

```bash
# Build normally
python -m visprobe.api.runner your_test_file.py

# View results (same as before)
streamlit run src/visprobe/cli/dashboard.py -- your_test_file.py
```

The new structure will automatically apply to all test results.

---

## 🔮 Future Enhancements

Potential additions:
- Comparison dashboard (before/after retraining)
- Multi-test comparison (how different strategies compare)
- Benchmark comparison (is my robustness good?)
- Automated solution suggestions (ML-based recommendation generation)
- Export to training framework (direct integration with PyTorch Lightning, etc.)

---

## 💬 Feedback

This redesign is based on:
- User feedback from testing frameworks
- Academic research on interpretability
- Best practices from industry ML dashboards (Wandb, MLflow, etc.)

We'd love to hear what works and what doesn't!
