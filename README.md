# credit-card-fraud-detection
Machine learning project using logistic regression to detect fraudulent credit card transactions with visualizations and a Streamlit dashboard.
# 🛡️ Credit Card Fraud Detection

This project uses a real-world dataset to train a machine learning model that detects fraudulent credit card transactions. It includes data processing, class balancing, model training, and an interactive Streamlit dashboard for visualization.

---

## 📊 Dataset Overview

- 📁 Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 💳 Transactions: 284,807
- ⚠️ Fraud Cases: 492 (only ~0.17%)

To avoid upload limits, only a 10,000-row sample (`creditcard_sample.csv`) is included.  
Please download the full dataset directly from Kaggle if needed.

---

## ✅ Key Features

- 🧹 Cleaned and preprocessed data using pandas
- 📉 Handled extreme class imbalance via under-sampling
- 🤖 Trained a Logistic Regression model on a balanced dataset
- 🎯 Achieved:
  - **94% Accuracy**
  - **83% Precision**
  - ROC AUC Score ≈ 0.95
- 📈 Visualized results with:
  - Confusion matrix
  - ROC curve
  - Class and amount distributions
- 🖥️ Built an interactive Streamlit dashboard

---

## 🚀 Run This Project Locally

### ▶️ Step 1: Install Required Packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit
