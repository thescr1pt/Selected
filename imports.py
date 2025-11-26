from sklearn.datasets import fetch_openml, load_iris, load_diabetes

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder

from sklearn.linear_model import SGDRegressor, SGDClassifier, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

import matplotlib.pyplot as plt
import numpy as np