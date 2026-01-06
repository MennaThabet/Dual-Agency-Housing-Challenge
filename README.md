# Dual-Agency Housing Challenge 🏠  
**2nd Place Solution — Regression with Distribution Shift Handling**

## Overview
This repository contains my solution for the **APPLAI ’26 Dual-Agency Housing Challenge**, where the goal was to predict **median house values** from two heterogeneous data sources with **severe distribution shifts**.

The focus of this work was not only model accuracy, but also **robust data preprocessing**, **distribution-shift detection**, and **reproducible ML engineering**.  
The final solution achieved **2nd place on the private leaderboard**.

---

## Key Contributions
- Detected and analyzed **distribution shifts** between merged data sources.
- Designed custom preprocessing transformers to handle:
  - Unit and scale mismatches
  - Bimodal feature distributions
  - Outliers and extreme values
- Built a **K-Fold neural network training pipeline** with ensembling.
- Saved preprocessing artifacts and trained models for **full reproducibility**.
- Performed systematic experimentation to improve generalization.

---

## Methodology

### 1. Data Exploration & Shift Detection
- Exploratory data analysis to compare feature distributions across agencies.
- Identification of inconsistent units and bimodal behaviors in multiple features.

### 2. Preprocessing Pipeline
Custom preprocessing components were implemented to stabilize training:
- **UnitAligner**: Aligns features with inconsistent measurement units.
- **BimodalSplitter**: Separates and normalizes bimodal feature distributions.
- **Clipper**: Limits extreme outliers to reduce training instability.
- Robust scaling and missing-value handling.

All preprocessing steps are fitted on training data only and saved as artifacts.

### 3. Model Training
- Neural networks built with **TensorFlow / Keras**.
- **K-Fold cross-validation** to reduce variance.
- Ensembling across folds for final predictions.
- Evaluation using **Mean Squared Error (MSE)**.

### 4. Experiments
- Baseline vs enhanced preprocessing
- Log-transformed targets
- Loss function variations
- Seed-based ensembling
- Stacking and averaging strategies

---

## Results
- **Private Leaderboard Rank:** 🥈 2nd Place  
- **Private Score (MSE):** `0.23875`

---

## Technologies Used
- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Joblib**
- **NumPy / Pandas**
- **Git**

---

## Reproducibility
- Fixed random seeds for experiments.
- Saved preprocessing pipelines and trained models using `joblib`.
- Clear separation between training, validation, and inference steps.

---

## Notes & Limitations
- The solution is optimized for robustness under distribution shifts rather than minimal model complexity.
- Neural networks were chosen for flexibility; tree-based models were explored but underperformed under shifts.
- Further improvements could include uncertainty estimation and automated shift detection.

---

## Author
**Menna Thabet**  
Computer Science & AI Student  

## Repository Structure
