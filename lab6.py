import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. Load Adult Dataset
# ---------------------------------------------------------
data = fetch_openml("adult", version=2, as_frame=True)
df = data.frame

# Drop missing values
df = df.dropna()

# Target variable
y = df["class"]
X = df.drop("class", axis=1)

# ---------------------------------------------------------
# 2. Separate categorical & continuous features
# ---------------------------------------------------------
cont_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns

X_cont = X[cont_cols]
X_cat = X[cat_cols]
# ---------------------------------------------------------
# 3. Split into train & test
# ---------------------------------------------------------
(X_cont_train, X_cont_test,X_cat_train, X_cat_test,y_train, y_test) = train_test_split(X_cont, X_cat, y,test_size=0.3, random_state=42)

# ---------------------------------------------------------
# 4. Encode categorical & scale continuous
# ---------------------------------------------------------
encoder = OrdinalEncoder()
X_cat_train_encoded = encoder.fit_transform(X_cat_train)
X_cat_test_encoded = encoder.transform(X_cat_test)

scaler = StandardScaler()
X_cont_train_scaled = scaler.fit_transform(X_cont_train)
X_cont_test_scaled = scaler.transform(X_cont_test)

# ---------------------------------------------------------
# 5. Train Naive Bayes Models
# ---------------------------------------------------------
gnb = GaussianNB()
gnb.fit(X_cont_train_scaled, y_train)

cnb = CategoricalNB(alpha=1.0)
cnb.fit(X_cat_train_encoded, y_train)

# ---------------------------------------------------------
# 6. Predict probabilities & combine
# ---------------------------------------------------------
proba_cont = gnb.predict_proba(X_cont_test_scaled)
proba_cat = cnb.predict_proba(X_cat_test_encoded)
eps = 1e-12
log_proba_total = np.log(proba_cont + eps) + np.log(proba_cat)

y_pred_idx = np.argmax(log_proba_total, axis=1)
y_pred = gnb.classes_[y_pred_idx]

# ---------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"Combined Mixed NB Accuracy: {accuracy:.4f}")

# New instance as DataFrame
new_instance = pd.DataFrame([{
    "age": 35,
    "fnlwgt": 200000,
    "education-num": 13,
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "workclass": "Private",
    "education": "Bachelors",
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "native-country": "United-States"
}])

# Split continuous and categorical
new_cont = new_instance[cont_cols]
new_cat = new_instance[cat_cols]

# Scale continuous features (same scaler)
new_cont_scaled = scaler.transform(new_cont)

# Encode categorical features (same encoder)
new_cat_encoded = encoder.transform(new_cat)
# Predict probabilities
proba_cont_new = gnb.predict_proba(new_cont_scaled)
proba_cat_new = cnb.predict_proba(new_cat_encoded)

# Combine log-probabilities safely
eps = 1e-12
log_proba_total_new = np.log(proba_cont_new + eps) + np.log(proba_cat_new)

# Choose class with highest probability
y_pred_idx_new = np.argmax(log_proba_total_new, axis=1)
y_pred_new = gnb.classes_[y_pred_idx_new]

print(f"Predicted class for the new instance: {y_pred_new[0]}")
