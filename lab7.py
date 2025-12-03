import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# ----------------------
# 1. Analyze the dataset
# ----------------------
iris = datasets.load_iris()
X = iris.data # Features (4 per sample)
y = iris.target # Target labels (0,1,2)
print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
# ----------------------
# 2. Data preprocessing
# ----------------------
# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42, stratify=y
)
# Scale features
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(X_train)
Xtest_scaled = scaler.transform(X_test)
# ----------------------
# 3. Fit the RBF SVM model
# ----------------------
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(Xtrain_scaled, y_train)
# ----------------------
# 4. Evaluate the fitted model
# ----------------------
y_pred = rbf_svm.predict(Xtest_scaled)
print(f"RBF SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# ----------------------
# 5. Generate predictions for new data
# ----------------------
new_samples = np.array([
 [5.1, 3.5, 1.4, 0.2],
 [6.2, 3.4, 5.4, 2.3],
 [5.9, 3.0, 4.2, 1.5]
])
# Scale new data
new_samples_scaled = scaler.transform(new_samples)
# Predict classes
new_preds = rbf_svm.predict(new_samples_scaled)
target_names = iris.target_names
pred_class_names = target_names[new_preds]
print("Predictions for new samples:", pred_class_names)