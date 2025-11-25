import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score
import numpy as np

# ===========================
# Load dataset
# ===========================
iris = load_iris()
X = iris.data
y = iris.target

# ===========================
# 3-way split: 60% train / 20% val / 20% test
# ===========================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# ===========================
# Standardize features
# ===========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ===========================
# Hyperparameters
# ===========================
learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
n_epochs = 50

best_lr = None
best_loss = float('inf')
best_train_loss_history = []
best_val_loss_history = []

# ===========================
# Training loop for LR search
# ===========================
for lr in learning_rates:
    clf = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=lr, random_state=42)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(n_epochs):
        clf.partial_fit(X_train, y_train, classes=np.unique(y_train))

        # --- Training performance ---
        y_train_prob = clf.predict_proba(X_train)
        y_train_pred = clf.predict(X_train)
        # calculate the training loss and accuracy
        train_loss = log_loss(y_train, y_train_prob)
        train_acc = accuracy_score(y_train, y_train_pred)

        # --- Validation performance ---
        y_val_prob = clf.predict_proba(X_val)
        y_val_pred = clf.predict(X_val)
        # calculate the validation loss and accuracy
        val_loss = log_loss(y_val, y_val_prob)
        val_acc = accuracy_score(y_val, y_val_pred)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

    print(f"Learning rate {lr:.4f} → Final Val Log Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # Track best learning rate
    if val_loss < best_loss:
        best_loss = val_loss
        best_lr = lr
        best_train_loss_history = train_loss_history
        best_val_loss_history = val_loss_history

print(f"\n Best learning rate: {best_lr} with Validation Log Loss: {best_loss:.4f}")

# ===========================
# Plot: Log Loss Curves
# ===========================
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs + 1), best_train_loss_history, label="Training Log Loss (MLE)")
plt.plot(range(1, n_epochs + 1), best_val_loss_history, label="Validation Log Loss (MLE)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy (Log Loss)")
plt.title(f"Training vs Validation Log Loss (Best lr={best_lr})")
plt.legend()
plt.grid(True)
plt.show()

# ===========================
# Retrain using Train + Validation (Full Training Data)
# ===========================
X_full = np.concatenate((X_train, X_val), axis=0)
y_full = np.concatenate((y_train, y_val), axis=0)


clf_best = SGDClassifier(
    loss='log_loss',
    learning_rate='constant',
    eta0=best_lr,
    random_state=42
)
clf_best.fit(X_full,y_full)

# ===========================
# Final Evaluation on Test Set
# ===========================
y_test_pred = clf_best.predict(X_test)
y_test_prob = clf_best.predict_proba(X_test)
test_loss = log_loss(y_test, y_test_prob)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n Final Model Evaluation on Test Set:")
print(f"  Log Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.4f}")