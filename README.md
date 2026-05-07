# Model-Prediction-Task-# 
🍽️ Restaurant Revenue Prediction 
### Predictive Modeling using Machine Learning

This repository contains a machine learning project designed to predict restaurant revenue using operational and financial data. By applying advanced regression techniques, we identify the key factors that drive a restaurant's success.

---

## 📊 Project Overview
The objective was to build a supervised learning model capable of estimating annual revenue. The project involved cleaning a "dirty" dataset, performing feature engineering on categorical variables, and training a high-precision model.

### 🔑 Key Features
* **Supervised Learning:** Applied Random Forest Regression to capture complex, non-linear relationships.
* **Data Wrangling:** Preprocessed "dirty" data, handled missing values, and encoded categorical features like `Location` and `Cuisine`.
* **Evaluation:** Used R-Squared and Mean Absolute Error (MAE) to validate model performance.
* **Data Visualization:** Created Actual vs. Predicted plots and Feature Importance charts.

---

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
* **Environment:** VS Code / Terminal

---

## 📈 Model Performance
The model achieved exceptional accuracy, demonstrating a near-perfect fit for the provided dataset.

| Metric | Result |
| :--- | :--- |
| **R-Squared Score** | **0.9992** |
| **Mean Absolute Error (MAE)** | **$5,800.72** |

### **Actual vs. Predicted Revenue**
The model's predictions follow the identity line almost perfectly, signifying extremely high reliability:
![Actual vs Predicted Plot](image_c4bfcf.png)

---

## 🧪 Dataset Insights
The model analyzed the following features to determine revenue:
* **Operational:** Seating Capacity, Average Meal Price, Chef Experience.
* **Marketing:** Budget, Social Media Followers.
* **Customer Feedback:** Rating, Number of Reviews, Service Quality.
* **Amenities:** Parking Availability, Weekend/Weekday Reservations.

---

## 🚀 How to Use
1. **Clone the Repo:**
   ```bash

   Install Dependencies:

Bash
pip install pandas scikit-learn matplotlib seaborn
Run the Model:

Bash
python model.py
💡 Conclusion
The Random Forest Regressor successfully decoded the relationship between restaurant features and income. The analysis reveals that factors like Average Meal Price and Seating Capacity are the strongest predictors of annual revenue.
   git clone [https://github.com/ABIVARMAN1/Thiranex-task2/tree/main]
   Created by ABIVARMAN
