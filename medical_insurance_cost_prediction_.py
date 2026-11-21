# Medical Insurance Cost Prediction
# Goal: predict a person's insurance charges using Linear, Polynomial and Ridge Regression.
# Tip: each chart opens in a window. Close the window to continue the program.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Folder to save the charts (used later in the README)
os.makedirs("images", exist_ok=True)

# ---------------------------------------------------------
# Step 1: Load the data
# ---------------------------------------------------------
url = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv"
)

# The file has no column names, so we add them ourselves
column_names = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df = pd.read_csv(url, header=None, names=column_names)

print(df.head())

# ---------------------------------------------------------
# Step 2: Clean the data
# ---------------------------------------------------------
# The file uses "?" for missing values, so change them to NaN
df = df.replace("?", np.nan)

# Make sure every column is a number
df = df.apply(pd.to_numeric, errors="coerce")

print("\nMissing values:")
print(df.isna().sum())

# Missing smoker -> use the most common value
most_common_smoker = df["smoker"].mode()[0]
df["smoker"] = df["smoker"].fillna(most_common_smoker).astype(int)

# Missing age -> use the average age
df["age"] = df["age"].fillna(df["age"].mean()).round().astype(int)

# Remove any rows that still have missing values
df = df.dropna()

# Round charges to 2 decimal places
df["charges"] = df["charges"].round(2)

# ---------------------------------------------------------
# Step 3: Explore the data
# ---------------------------------------------------------
print("\nCorrelation between columns:")
print(df.corr().round(2))

# Chart 1: BMI vs charges
sns.regplot(x="bmi", y="charges", data=df)
plt.ylim(0, None)
plt.title("BMI vs Insurance Charges")
plt.savefig("images/insurance_bmi_vs_charges.png", dpi=150)
plt.show()

# Chart 2: charges for non-smokers vs smokers
sns.boxplot(x="smoker", y="charges", data=df)
plt.xlabel("Smoker (0 = no, 1 = yes)")
plt.ylabel("Charges")
plt.title("Insurance Charges by Smoking Status")
plt.savefig("images/insurance_charges_by_smoker.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# Step 4: Prepare the data for the models
# ---------------------------------------------------------
# Region is a category (1, 2, 3, 4), not a size. Region 4 is not "bigger" than region 1.
# get_dummies changes it into separate yes/no columns so the model treats it correctly.
df = pd.get_dummies(df, columns=["region"], prefix="region", drop_first=True, dtype=float)

# X = information used to predict, y = what we want to predict
X = df.drop(columns=["charges"])
y = df["charges"]

# ---------------------------------------------------------
# Step 5: Split into training and test sets
# The model learns from the training data (80%).
# We then check it on the test data (20%) that it has never seen.
# ---------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("\nTraining samples:", len(x_train))
print("Test samples:", len(x_test))

# ---------------------------------------------------------
# Step 6: Build and test the models
# R2 = how well the model fits (closer to 1 is better)
# RMSE = average error in dollars (smaller is better)
# ---------------------------------------------------------
print("\n--- Model Results (on test data) ---")

# Model 1: Simple Linear Regression using smoker only
model_1 = LinearRegression()
model_1.fit(x_train[["smoker"]], y_train)
pred_1 = model_1.predict(x_test[["smoker"]])
print("1. Linear (smoker only)  R2:", round(r2_score(y_test, pred_1), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_1) ** 0.5))

# Model 2: Multiple Linear Regression using all columns
model_2 = LinearRegression()
model_2.fit(x_train, y_train)
pred_2 = model_2.predict(x_test)
print("2. Linear (all features) R2:", round(r2_score(y_test, pred_2), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_2) ** 0.5))

# Model 3: Polynomial Regression (scale the data, add polynomial terms, then fit)
model_3 = Pipeline([
    ("scale", StandardScaler()),
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression()),
])
model_3.fit(x_train, y_train)
pred_3 = model_3.predict(x_test)
print("3. Polynomial            R2:", round(r2_score(y_test, pred_3), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_3) ** 0.5))

# Model 4: Ridge Regression
model_4 = Ridge(alpha=0.1)
model_4.fit(x_train, y_train)
pred_4 = model_4.predict(x_test)
print("4. Ridge                 R2:", round(r2_score(y_test, pred_4), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_4) ** 0.5))

# Model 5: Polynomial + Ridge Regression
model_5 = Pipeline([
    ("scale", StandardScaler()),
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", Ridge(alpha=0.1)),
])
model_5.fit(x_train, y_train)
pred_5 = model_5.predict(x_test)
print("5. Polynomial + Ridge    R2:", round(r2_score(y_test, pred_5), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_5) ** 0.5))

# ---------------------------------------------------------
# Step 7: Chart comparing the models
# ---------------------------------------------------------
model_names = ["Smoker only", "All features", "Polynomial", "Ridge", "Poly + Ridge"]
r2_scores = [
    r2_score(y_test, pred_1), r2_score(y_test, pred_2), r2_score(y_test, pred_3),
    r2_score(y_test, pred_4), r2_score(y_test, pred_5),
]

plt.figure(figsize=(9, 4))
plt.barh(model_names, r2_scores, color="blue")
plt.xlabel("Test R2 (higher is better)")
plt.title("Insurance Charges - Model Comparison")
plt.gca().invert_yaxis()
plt.savefig("images/insurance_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
