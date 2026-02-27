# 🏦 Retail Bank Customer Segmentation & Profitability Prediction

A machine learning project that segments bank customers using **K-Means clustering** and predicts customer profitability using **Logistic Regression**, deployed with an interactive **Streamlit** web UI.

---

## 📌 Project Overview

This project analyses bank transaction data to:

1. **Segment customers** into distinct groups based on their age, account balance, and transaction behaviour (unsupervised learning — K-Means).
2. **Predict** whether a customer is **High Profit** or **Standard** (supervised learning — Logistic Regression).
3. **Evaluate** model performance via accuracy metrics, confusion matrix, ROC curve, and classification report.
4. **Deploy** the pipeline as an interactive Streamlit web application for real-time predictions and data exploration.

---

## 🚀 Features

| Page | What it does |
|---|---|
| **🏠 Overview** | Key metrics (total records, features, clusters, accuracy), dataset preview and summary statistics |
| **📊 EDA** | Age distribution, account balance distribution, transaction amount distribution, gender split pie chart, correlation heatmap |
| **🧩 Clustering** | K-Means cluster statistics, interactive scatter plots with selectable axes, per-cluster breakdowns |
| **🔮 Predict** | Enter customer details (age, gender, balance, transaction amount) and get a live High Profit / Standard prediction with probability |
| **📈 Model Evaluation** | Accuracy & AUC metrics, confusion matrix heatmap, ROC curve, full classification report |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web UI
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualisation
- **Scikit-learn** — K-Means clustering, Logistic Regression, StandardScaler, model evaluation

---

## 📂 Project Structure

```
banksegmentation/
├── app.py                          # Streamlit application
├── BankSegmentation_Group6.ipynb   # Original Jupyter notebook
├── bank_transactions.csv           # Dataset (auto-generated if missing)
├── README.md                       # This file
```

---

## ⚙️ Setup & Run

### 1. Install dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

> **Note:** If `bank_transactions.csv` is not found in the project directory, the app automatically generates a synthetic fallback dataset for demonstration purposes.

---

## 🧪 ML Pipeline

### Data Preprocessing
- Parse & clean dates, extract customer **Age**
- Encode **Gender** (M → 1, F → 0)
- Fill missing **Account Balance** and **Transaction Amount** with median values

### K-Means Clustering (Unsupervised)
- Features: `Age`, `CustAccountBalance`, `TransactionAmount (INR)`
- Standard-scaled before clustering
- **k = 3** clusters

### Logistic Regression (Supervised)
- Target: `HighProfit` — 1 if `CustAccountBalance > ₹20,00,000`, else 0
- Features: `Age`, `CustGender`, `CustAccountBalance`, `TransactionAmount (INR)`
- 80/20 train-test split (stratified)
- Class-weight balanced to handle imbalanced data

---

## 👥 Team — Group 6

| Role | Responsibility |
|---|---|
| **A** | Data Preparation & Exploratory Data Analysis |
| **S** | K-Means Clustering & Streamlit Deployment & UI & Documentation  |
| **AS** | Logistic Regression Prediction & Business Risk Evaluation |
| **Y** | Testing |

---

## 📄 License

This project is for academic / educational purposes.
# Bank-Customer-Segmentation-Profitability-Prediction
