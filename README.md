# Pakistan Used Car Price Prediction System

A complete machine learning–based system to predict used car prices in Pakistan.
The project covers data preprocessing, feature engineering, model training, and
deployment using Streamlit.

---

##  Project Structure

├── data/
│   ├── pakwheels_used_car_data_v02.csv
│   └── cleaned_car_data.csv
│
├── models/
│   ├── best_model.pkl
│   └── features.pkl
│
├── preprocessing.ipynb
├── modeltraining.ipynb
├── app.py
├── requirements.txt
└── README.md

---

## Features

- Predicts realistic used car prices
- Handles categorical & numerical features
- Advanced feature engineering (age, engine category, brand class)
- Brand & Model price trend analysis
- Streamlit interactive dashboard

---

## Machine Learning Pipeline

- Data cleaning & preprocessing
- Feature engineering
- ColumnTransformer with:
  - OneHotEncoding (categorical)
  - Scaling (numerical)
- Log transformation on target price
- Random Forest Regressor
- End-to-end pipeline saved and reused in app

---
- Models Trained:

Random Forest Regressor
XGBoost Regressor (optional)
LightGBM Regressor (optional)

-T he best model is automatically selected based on R² score.
- Whichever had highest R² was saved as best_model.pkl

##  How to Run

###  Install dependencies
```bash
pip install -r requirements.txt
