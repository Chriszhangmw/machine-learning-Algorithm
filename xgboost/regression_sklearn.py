import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

dataset_train = './Boston_data/train.csv'
dataset_test = './Boston_data/test.csv'
data_train = pd.read_csv(dataset_train)
data_test = pd.read_csv(dataset_test)
X = data_train.drop(['ID', 'medv'], axis=1)
y = data_train.medv

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='reg:gamma')
model.fit(X_train, y_train)

ans = model.predict(X_test)
plot_importance(model)
plt.show()


