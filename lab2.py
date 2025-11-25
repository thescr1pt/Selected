import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import SGDRegressor 
from sklearn.metrics import mean_squared_error 


# ======================================= 
# Load and prepare data 
# ======================================= 
diabetes = datasets.load_diabetes() 
X = diabetes.data[:, 2:]  # Use features from BMI to the last 
y = diabetes.target 

# Split 80% training and 20% testing with random state 42 
X_train, X_test, y_train, y_test =  train_test_split(X,y,random_state=42, test_size=0.2)                               #Q1
# Scale features and target (Unbounded, continuous features) 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)                                                         #Q2
X_test = scaler.transform(X_test) 
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test =  scaler.transform(y_test.reshape(-1, 1)).ravel()                                                         #Q3


# ======================================= 
# Settings 
# ======================================= 
epochs = 100 
learning_rates = [0.0001, 0.001] 
avg_losses = {} 

# ======================================= 
# Training with different learning rates 
# ======================================= 
for lr in learning_rates: 
  model = SGDRegressor(eta0=lr, learning_rate="constant", random_state=42) 
  losses = [] 
  for epoch in range(epochs): 
    #fit the model                                                #Q4
    model.partial_fit(X_train, y_train)
    #predict the model                                            #Q5
    y_pred = model.predict(X_train)
    #calculate MSE 
    mse =  mean_squared_error(y_train, y_pred)                          #Q6
    losses.append(mse) 
  avg_losses[lr] = losses 

# ======================================= 
# Plot MSE vs. Epoch for each learning rate 
# ======================================= 
plt.figure(figsize=(10, 6)) 
for lr, losses in avg_losses.items(): 
  plt.plot(range(1, epochs + 1), losses, label=f"η = {lr}") 
  plt.xlabel("Epoch") 
  plt.ylabel("Training MSE (Loss)") 
  plt.title("Learning Rate Comparison — MSE vs. Epochs") 
  plt.legend(title="Learning Rate") 
  plt.show()

# ======================================= 
# Plot Regression Slopes (after training with best learning rate) 
# ======================================= 
best_lr = 0.001 

# Retrain model on all data with best learning rate 
final_model = SGDRegressor( 
max_iter=1, 
eta0=best_lr, 
learning_rate="constant", 
random_state=42 
) 


final_model.fit(X_train, y_train) 
y_test_pred = final_model.predict(X_test) 
overall_mse = mean_squared_error(y_test, y_test_pred) 
print(f"Best Learning Rate: {best_lr}") 
print(f"Overall Test MSE: {overall_mse:.4f}") 


# Get feature names (from BMI to the end) 
features = diabetes.feature_names[diabetes.feature_names.index("bmi"):] 
# Plot coefficients (slopes) 
plt.figure(figsize=(8, 5)) 
plt.bar(features, final_model.coef_)                                                        #Q7                                       
plt.axhline(0, color="black", linewidth=1) #AXes Horizontal Line 
plt.ylabel("Slope (Coefficient)") 
plt.title(f"Regression Slopes per Feature — η = {best_lr}") 
plt.show() 


#calculate Residuals 
residuals = y_test - y_test_pred                                                      #Q8
plt.figure(figsize=(8, 5))                                        #Q9
plt.scatter(y_test_pred, residuals, alpha=0.7) 
# flatten to 1D 
plt.axhline(0, color="red", linestyle="--", linewidth=1)  # baseline at 0
plt.xlabel("Predicted Values") 
plt.ylabel("Residuals (Actual - Predicted)") 
plt.title("Residuals vs Predicted") 
plt.show() 