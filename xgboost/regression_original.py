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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1, max_depth=8,
alpha = 8, n_estimators=500, reg_lambda=1)
xg_reg.fit(X_train, y_train)
x_test = data_test.drop(['ID'], axis=1)
predictions = xg_reg.predict(x_test)
ID = (data_test.ID).astype(int)
result = np.c_[ID, predictions]
np.savetxt('./' + 'xgb_submission.csv', result, fmt="%d,%.4f" ,header='ID,medv', delimiter=',', comments='')

plot_importance(xg_reg)
plt.show()


