#1. import the missing classes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.datasets import load_iris, fetch_california_housing

# ================================
#  1. CLASSIFICATION MODEL (IRIS)
# ================================
#2. import the missing classes
iris = load_iris()
x, y = iris.data, iris.target

#3. Split into train 70% / val 15% / test 15%
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

depths = range(1, 15)
train_loss_history, val_loss_history = [], []

#4. Train multiple Decision Tree Classifiers
for d in depths :
    model = DecisionTreeClassifier(max_depth=d, random_state=42,criterion='entropy')
    model.fit(x_train,y_train)

    # Predictions
    y_train_pred_proba = model.predict_proba(x_train)
    y_val_pred_proba = model.predict_proba(x_val)

    # Log loss for training & validation
    train_loss_history.append(log_loss(y_train, y_train_pred_proba))
    val_loss_history.append(log_loss(y_val, y_val_pred_proba))

#5. Plot the complexity curve
plt.figure(figsize=(8,5))
plt.plot(depths, train_loss_history, marker='o', label='Train Loss')
plt.plot(depths, val_loss_history, marker='s', label='Validation Loss')
plt.xlabel('Tree Depth')
plt.ylabel('Log Loss')
plt.title('Classification Decision Tree (Complexity Curve) Visualization')
plt.legend()
plt.show()

#6. Choose the best depth (lowest validation loss)
best_depth_class = depths[np.argmin(val_loss_history)]
print(f"Best depth for classification: {best_depth_class}")

#7. Retrain model ONLY on training data with best depth
clf_final = DecisionTreeClassifier(max_depth=best_depth_class, random_state=42, criterion='entropy')
clf_final.fit(x_train, y_train)

#8. Evaluate final model
y_test_proba = clf_final.predict_proba(x_test)
print(f"Final Classification Loss on Test Set: {log_loss(y_test, y_test_proba):.3f}")

tree_rules = export_text(clf_final, feature_names=iris.feature_names)
print(tree_rules)

plt.figure(figsize=(12, 8))
plot_tree(clf_final, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()


# ==================================
#  2. REGRESSION MODEL (CALIFORNIA)
# ==================================
housing = fetch_california_housing()
x, y = housing.data, housing.target

#split the data
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

depths = range(1, 15)
train_loss_history, val_loss_history = [], []

for d in depths :
    model = DecisionTreeRegressor(max_depth=d, random_state=42)
    model.fit(x_train, y_train)

    # Predictions
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    # Mean Squared Error (Loss)
    train_loss_history.append(mean_squared_error(y_train, y_train_pred))
    val_loss_history.append(mean_squared_error(y_val, y_val_pred))

# Plot the complexity curve
plt.figure(figsize=(8,5))
plt.plot(depths, train_loss_history, marker='o', label='Train Loss')
plt.plot(depths, val_loss_history, marker='s', label='Validation Loss')
plt.xlabel('Tree Depth')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Regression Decision Tree (Complexity Curve) Visualization')
plt.legend()
plt.show()

# Choose the best depth
best_depth_reg = depths[np.argmin(val_loss_history)]
print(f"Best depth for regression: {best_depth_reg}")

# Retrain final regression model only on training data
reg_final = DecisionTreeRegressor(max_depth=d, random_state=42)
reg_final.fit(x_train, y_train)


# Evaluate on test set
y_test_pred = reg_final.predict(x_test)
print(f"Final Regression MSE on Test Set: {mean_squared_error(y_test, y_test_pred):.3f}")
