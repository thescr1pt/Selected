import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = load_diabetes()
# Use only one feature (BMI feature at index 2)
diabetes_X = diabetes.data[:, 2:3]
#select only the target column
diabetes_Y = diabetes.target
# Split the data into training/testing sets (80% train, 20% test) with a

diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, diabetes_Y, random_state=9, test_size=0.2)

# Create linear regression object
regr = LinearRegression()
#Fit the model
regr.fit(diabetes_X_train,diabetes_y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
# The coefficients
print("Coefficient (slope):", regr.coef_[0])
print("Coefficient (Intercept):", regr.intercept_)
# The mean squared error
print(f"Mean squared error: {mean_squared_error(diabetes_y_test,diabetes_y_pred):.2f}")
# ==========================
# Plot outputs
# ==========================
plt.figure(figsize=(8,6))
# 1. Actual points
plt.scatter(diabetes_X_test, diabetes_y_test, color="black", label="Actualdata", s=80)
# 2. Regression line (connect predicted values)
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=2, label="Regression line")
# 3. Predicted points
plt.scatter(diabetes_X_test, diabetes_y_pred, color="blue", s=60, label="Predicted points")
# 4. Residuals (vertical dashed red lines)
plt.vlines(diabetes_X_test.ravel(), diabetes_y_pred, diabetes_y_test, color="red", linestyle="dashed", label="Residuals")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression")
plt.title("Simple Linear Regression on Diabetes Dataset (BMI Feature)")
plt.legend()
# plt.grid(True)
plt.show()