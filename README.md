# 📊 Customer Churn Prediction

A machine learning based web application to predict if a telecom customer is likely to churn based on his account details and service usage.

---

## 🚀 Project Overview

Churn among clients is the major problem in the telecommunication sector. In this project, machine learning techniques will be employed to recognize potential client churners, enabling organizations to engage in preventive measures to retain such clients.

The solution includes:
- Comprehensive exploratory data analysis (EDA) to uncover customer churn trends
- Data preprocessing and feature engineering for improved model performance
- Machine learning model training with hyperparameter tuning and evaluation
- A Flask-based web application that provides real-time churn predictions

---

## 🗂️ Project Structure

```
Customer_Churn_Prediction/
├── app.py                  # Flask backend
├── churn_analysis.ipynb    # Jupyter notebook (EDA + model training)
├── best_model.pkl          # Saved Random Forest model
├── encoder.pkl             # Saved label encoders
├── scaler.pkl              # Saved standard scaler
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Frontend form
├── static/
│   └── style.css           # Styling
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 📁 Dataset

- **Name:** Telco Customer Churn
- **Source:** [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers, 21 features
- **Target:** `Churn` (Yes / No)

---

## 🧠 Approach

1. **Data Cleaning** — Cleaned the dataset by converting the `TotalCharges` column to numeric format and removing the unnecessary `customerID` column
2. **EDA** — Performed exploratory data analysis (EDA) to study customer distributions, feature relationships, and churn behavior
3. **Encoding** — Encoded categorical features using `LabelEncoder`
4. **Scaling** — Scaled numerical features such as `tenure`, `MonthlyCharges`, and `TotalCharges` using `StandardScaler`
5. **Class Imbalance** — Handled class imbalance with `SMOTE`, balancing churn and non-churn classes
6. **Model Training** — Trained and optimized `Random Forest` and `XGBoost` models using `GridSearchCV` with 5-fold cross-validation
7. **Best Model** — Selected Random Forest as the best-performing model with around 78% accuracy and 0.74 ROC-AUC score
8. **Deployment** — Deployed the model through a Flask web application for real-time customer churn prediction

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Gowrykrishna/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask app
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

---

## 📦 Dependencies

```
matplotlib
seaborn
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
flask

```

Install all at once:
```bash
pip install flask pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
```

---

## 🖥️ How to Use

1. Open the web app at `http://127.0.0.1:5000`
2. Fill in the customer details in the form
3. Click **Predict Churn**
4. The result shows whether the customer will churn along with the probability

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | 78% | 0.74 |
| XGBoost | 82% (CV) | — |

---
