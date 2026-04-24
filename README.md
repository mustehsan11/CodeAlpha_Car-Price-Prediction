# 🚗 Car Price Prediction with Machine Learning

> **Data Science Internship — Task 3**  
> A regression-based machine learning project to predict used car selling prices using features like brand, present price, mileage, fuel type, and age.

---

## 📌 Overview

This project builds and compares multiple regression models to predict the **resale price of used cars** in the Indian market. It covers the complete ML pipeline — data exploration, feature engineering, model training, evaluation, and deployment-ready prediction — using a real-world dataset of 301 car listings.

---

## 📁 Project Structure

```
car-price-prediction/
│
├── Car_Price_Prediction.ipynb   # Main Colab notebook
├── car_data.csv                 # Dataset — 301 used car listings
└── README.md
```

---

## 📂 Dataset

| Feature | Type | Description |
|---------|------|-------------|
| `Car_Name` | Categorical | Model name of the car |
| `Year` | Numerical | Year of manufacture |
| `Selling_Price` | Numerical | **Target** — resale price in Lakhs (₹) |
| `Present_Price` | Numerical | Current showroom price in Lakhs (₹) |
| `Driven_kms` | Numerical | Total kilometres driven |
| `Fuel_Type` | Categorical | Petrol / Diesel / CNG |
| `Selling_type` | Categorical | Dealer / Individual |
| `Transmission` | Categorical | Manual / Automatic |
| `Owner` | Numerical | Number of previous owners |

**Records:** 301 rows, 9 columns, no missing values  
**Price range:** ₹0.10 – ₹35.00 Lakhs

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | ML models, preprocessing, evaluation |

---

## 🔄 Workflow

```
Load Data → EDA → Feature Engineering → Train/Test Split
    → Train 6 Models → Evaluate & Compare → Best Model Analysis
        → Cross-Validation → Predict New Car Price
```

---

## ⚙️ Feature Engineering

| Transformation | Description |
|---------------|-------------|
| `Car_Age = 2024 - Year` | More meaningful than raw year — captures depreciation |
| Label Encoding | Fuel Type, Selling Type, Transmission → numeric |
| StandardScaler | Applied to features for linear models |
| Drop `Car_Name` & `Year` | Removed after feature extraction |

---

## 🤖 Models Trained & Compared

| Model | Type | Notes |
|-------|------|-------|
| Linear Regression | Linear | Baseline model |
| Ridge Regression | Linear + L2 | Controls overfitting |
| Lasso Regression | Linear + L1 | Feature selection via sparsity |
| Decision Tree | Non-linear | Interpretable tree structure |
| **Random Forest** | Ensemble | **Best performer** |
| Gradient Boosting | Ensemble | Strong runner-up |

---

## 📊 Visualisations Included

| # | Chart | Purpose |
|---|-------|---------|
| 1 | Price distribution (raw + log) | Understand skewness |
| 2 | Correlation heatmap | Feature relationships |
| 3 | Price by fuel type & transmission | Categorical impact on price |
| 4 | Scatter plots vs key features | Visual feature-price relationships |
| 5 | Model comparison bar charts | R², MAE, RMSE side by side |
| 6 | Actual vs Predicted scatter | Prediction accuracy visualised |
| 7 | Feature importance chart | What drives the model's decisions |
| 8 | Residual analysis | Check for prediction bias |

---

## 📈 Results

| Model | R² Score | MAE (₹ Lakhs) | RMSE (₹ Lakhs) |
|-------|----------|----------------|-----------------|
| **Random Forest** | **~0.97** | **~0.65** | **~1.20** |
| Gradient Boosting | ~0.96 | ~0.70 | ~1.30 |
| Decision Tree | ~0.93 | ~0.85 | ~1.80 |
| Ridge Regression | ~0.86 | ~1.35 | ~2.55 |
| Linear Regression | ~0.85 | ~1.40 | ~2.60 |
| Lasso Regression | ~0.84 | ~1.45 | ~2.70 |

> **Random Forest explains ~97% of the variance in used car prices.**

---

## 🔑 Key Findings

1. **Present_Price** is the single strongest predictor of resale value
2. **Car_Age** is the second most important feature — each year reduces price by ~₹0.4L on average
3. **Diesel cars** command a significant price premium over Petrol in India's used-car market
4. **Automatic transmission** cars are priced ~40% higher than Manual equivalents
5. **High mileage** (Driven_kms) negatively impacts price — ~₹0.3L drop per 10,000 km
6. The model generalises well — 5-fold cross-validation R² remains stable at ~0.96

---

## 🔮 Predict a New Car's Price

The notebook includes a ready-to-use prediction cell. Just fill in the car's details:

```python
new_car = pd.DataFrame({
    'Present_Price'   : [6.50],   # Showroom price in Lakhs
    'Driven_kms'      : [35000],  # Total km driven
    'Fuel_Type_enc'   : [1],      # 0=CNG, 1=Diesel, 2=Petrol
    'Selling_type_enc': [0],      # 0=Dealer, 1=Individual
    'Transmission_enc': [0],      # 0=Automatic, 1=Manual
    'Owner'           : [0],      # Previous owners
    'Car_Age'         : [6],      # Age in years
})

predicted_price = best_model.predict(new_car)[0]
print(f"Predicted Price: ₹ {predicted_price:.2f} Lakhs")
```

---

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Car_Price_Prediction.ipynb`
3. Run **Step 2** — a file upload button will appear
4. Upload `car_data.csv` when prompted
5. Run all cells (`Runtime → Run all`)

---

## 👤 Author

Syed Mustehsan Akhtar Kazmi 

Data Science Intern  
