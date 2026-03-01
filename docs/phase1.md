
---

# Nukefraud — Phase 1: Baseline Fraud Detection System

## Introduction

Financial fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent a tiny fraction of total activity. Traditional accuracy-based evaluation is insufficient in this setting due to extreme class imbalance.

**Nukefraud (Phase 1)** establishes a structured, production-oriented machine learning pipeline for credit card fraud detection with a strong emphasis on:

* Reproducible engineering practices
* Cost-aware decision optimization
* Modular system architecture
* Business-aligned evaluation metrics

This phase prioritizes engineering rigor and decision policy optimization over model complexity.

---

## Table of Contents

1. [Objectives](#objectives)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Project Architecture](#project-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Features](#features)
8. [Dependencies](#dependencies)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [Future Development](#future-development)
12. [Contributors](#contributors)
13. [License](#license)

---

## Objectives

Phase 1 was designed to:

1. Establish a clean and scalable project architecture
2. Implement a baseline fraud detection model
3. Properly handle extreme class imbalance
4. Optimize classification threshold using business cost modeling
5. Persist trained model artifacts for future API deployment

---

## Theoretical Foundations

### Logistic Regression

The baseline model uses **Logistic Regression**, modeling binary outcome probability as:

[
P(y=1 \mid x) = \frac{1}{1 + e^{-w^T x}}
]

Why logistic regression?

* Produces calibrated probabilities
* Interpretable coefficients
* Performs well on scaled tabular data
* Efficient and production-friendly

Class imbalance was addressed using:

```python
class_weight="balanced"
```

This adjusts the loss function to penalize minority class errors more heavily.

---

### Why Accuracy Is Misleading

Dataset characteristics:

* 284,807 total transactions
* 492 fraud cases (~0.17%)

A naive model predicting “Not Fraud” always would achieve ~99.8% accuracy.

Therefore, evaluation focuses on:

* ROC-AUC
* PR-AUC (more informative under imbalance)
* Precision
* Recall
* Expected Business Cost

---

### Precision–Recall Tradeoff

Fraud detection involves asymmetric costs:

* False Negative (missed fraud) → High financial loss
* False Positive (false alarm) → Operational review cost

---

## Project Architecture

### Folder Structure

```
nukefraud/
│
├── app/
├── data/
│   ├── raw/
│   │   ├── creditcard.csv
│   └── processed/
├── docs/
│   ├── phase1.md
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── thresholding.py
│   ├── config.py
│   └── models/
│       ├── baseline.py
│       └── trainer.py
│
├── tests/
├── README.md
├── requirements.txt
└── LICENSE
```

### Design Principles

* Separation of data logic and modeling
* `src/` structured as a Python package
* No notebook-dependent pipelines
* Clean artifact saving for future inference APIs

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nnfuad/nukefraud.git
cd nukefraud
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train Baseline Model

```bash
python -m src.models.trainer
```

### Model Output

The training process:

* Fits logistic regression with class balancing
* Optimizes threshold using cost-sensitive evaluation
* Saves trained model artifact into `models/` directory

---

## Results

### Default Threshold (0.5)

Confusion Matrix:

```
[[55478  1386]
 [    8    90]]
```

* Fraud Precision ≈ 0.06
* Fraud Recall ≈ 0.92
* Expected Cost ≈ 21,860

Problem:
Extremely high false positive rate → Operationally unrealistic.

---

### Cost-Based Threshold Optimization

Assumptions:

* False Negative cost = $1000
* False Positive cost = $10

Optimized threshold ≈ **0.97**

Confusion Matrix:

```
[[56769    95]
 [   11    87]]
```

* Fraud Precision ≈ 0.48
* Fraud Recall ≈ 0.89
* Expected Cost ≈ 11,950

**Business loss reduced by ~45%.**

This highlights the importance of separating:

* Probability estimation
* Decision policy

---

## Features

* Clean modular architecture
* Imbalance-aware training
* Cost-sensitive threshold optimization
* Saved model artifacts for inference
* Reproducible virtual environment setup
* Production-aligned evaluation metrics

---

## Dependencies

Core dependencies:

* numpy
* pandas
* scikit-learn
* torch
* fastapi
* streamlit
* joblib

See `requirements.txt` for full details.

---

## Configuration

Threshold cost assumptions are defined in:

```
src/config.py
```

You may adjust:

* False Negative cost
* False Positive cost
* Threshold search granularity

This allows adapting the system to different business contexts.

---

## Troubleshooting

### Virtual Environment Conflicts

**Issue:**
Mixed `conda` and `venv` environments caused missing module errors.

**Fix:**

* Deactivate conda
* Reinstall dependencies inside project `venv`
* Correct VS Code interpreter selection

---

### Python Package Import Errors

**Issue:**
Relative imports failed.

**Fix:**
Added:

```
src/__init__.py
src/models/__init__.py
```

Converted `src/` into a proper Python package.

---

### Threshold Evaluation Bug

**Issue:**
Model optimized threshold but evaluation used default 0.5.

**Fix:**
Evaluation explicitly updated to use optimized threshold.

---

## Future Development

### Modeling Improvements

* Deep MLP with weighted BCE loss
* Feature importance analysis
* Model calibration techniques
* Cross-validation

### System Engineering

* FastAPI inference endpoint
* Streamlit interactive dashboard
* Docker containerization
* Cloud deployment (Render free tier)

### Advanced Techniques

* Ensemble methods
* XGBoost comparison
* Anomaly detection models
* Real-time simulation pipeline

---

## Contributors

Currently maintained by the Nukefraud project author.

---

## License

This project is licensed under the **MIT License**.

---

## Phase 1 Conclusion

Phase 1 establishes:

* A clean engineering foundation
* A reproducible ML pipeline
* Cost-aware decision optimization
* Business-aligned evaluation

The system is now ready for expansion into deep learning modeling, API deployment, and production experimentation.

---