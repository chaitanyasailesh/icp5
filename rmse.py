from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# importing the data set
wh = pd.read_csv('wine.csv')

# searching for attributes which have null values
print(wh["quality"].isnull().any())

numeric_features  = wh.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print(corr['quality'].sort_values(ascending=False)[:4])
X_train, X_test = train_test_split(wh, test_size=0.2)
y_train = X_train['quality']

X_train = X_train.drop(columns=['quality'])
y_test = X_test['quality']
X_test = X_test.drop(columns=['quality'])
# create regression model and train it
reg = LinearRegression().fit(X_train, y_train)


prediction = reg.predict(X_test)
# evaluating the required models by metrics
mean_squared_error = mean_squared_error(y_test, prediction)
r2_score = r2_score(y_test, prediction)
# rmse(measures the differences between values predicted by a model)
# r2(regression score function)
print("mean squared error is :", mean_squared_error)
print("r2_score is: ", r2_score)