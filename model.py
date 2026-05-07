import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. LOAD DATA
# Ensure your filename matches exactly
df = pd.read_csv('restaurant_data.csv') 

# 2. PREPROCESSING
def preprocess_restaurant_data(df):
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle Categorical variables (convert text to numbers)
    # We'll use One-Hot Encoding for Location, Cuisine, and Parking Availability
    categorical_cols = ['Location', 'Cuisine', 'Parking Availability']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Drop the 'Name' column as it's unique to every row and doesn't help prediction
    if 'Name' in df.columns:
        df = df.drop('Name', axis=1)
        
    # Fill any missing values with the median (just in case)
    df = df.fillna(df.median(numeric_only=True))
    
    return df

df_cleaned = preprocess_restaurant_data(df)

# 3. SPLIT DATA
X = df_cleaned.drop('Revenue', axis=1)
y = df_cleaned['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN MODEL
# Random Forest is perfect for capturing how 'Chef Experience' and 'Marketing' interact
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. EVALUATE
y_pred = model.predict(X_test)

print("--- MODEL PERFORMANCE ---")
print(f"R-Squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):,.2f}")

# 6. VISUALIZE RESULTS
plt.figure(figsize=(10, 6))

# Plotting Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title('What Drives Revenue in Your Dataset?')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs. Predicted Revenue')
plt.show()