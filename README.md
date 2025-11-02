# 📊 Customer Churn Prediction

This project predicts whether a telecom customer is likely to **churn** (leave the company) using a **Random Forest Classifier** model.

It includes:
- `Project_final.ipynb` → Jupyter Notebook for training and saving the model  
- `app.py` → Streamlit app for making churn predictions  

## 🧠 Model Information
- Algorithm: `RandomForestClassifier`
- Dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Encoded categorical features using `LabelEncoder`
- No scaling applied (Random Forest is insensitive to feature scaling)
- Saved files:
  - `model.pkl`
  - `encoders.pkl`
  - `columns.pkl`

## ⚙️ Installation
Install the required libraries:
```bash
pip install streamlit pandas scikit-learn
