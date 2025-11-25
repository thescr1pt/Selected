# 1. import needed classes
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 2. Load dataset
iris = load_iris()
x = iris.data
y = iris.target
# 3. Split into train/test sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature scaling (important for KNN)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. Try different K values using cross-validation
k_values = range(1, 11)
cv_accuracy_values = []

print("Cross-validation accuracy for each K:")
for k in k_values: #complete the line
    knn = KNeighborsClassifier(n_neighbors=k)
    # 5-fold cross-validation
    cv_scores = cross_val_score(knn,x_train,y_train,scoring="accuracy", cv=5)
    mean_acc = np.mean(cv_scores) #complete the line
    cv_accuracy_values.append(mean_acc) #complete the line
    print(f"K={k}, CV Accuracy={mean_acc:.3f}")

# 6. Plot the elbow curve (CV Accuracy vs K)
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_accuracy_values, marker='o', linestyle='--', color='b') #complete the line
plt.title("Elbow Method (Cross-Validation Accuracy vs. K)")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Cross-Validation Accuracy")
plt.grid(True)
plt.show()

# 7. Choose the best K (highest CV Accuracy)
best_k = k_values[np.argmax(cv_accuracy_values)]
print(f"\nBest K found via CV: {best_k} with CV Accuracy={max(cv_accuracy_values):.3f}")

# 8. Retrain using the best K
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(x_train,y_train)

# 9. Final evaluation on unseen test data and print the accuracy
y_pred = final_knn.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy is {acc}")