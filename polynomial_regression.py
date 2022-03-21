from itertools import count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import missingno as msno
from math import sqrt

plt.style.use('seaborn')

TRAIN_FILE_PATH = 'data/train.csv'
X = pd.read_csv(TRAIN_FILE_PATH, parse_dates=['datetime'])

X['year'] = X['datetime'].dt.year
X['month'] = X['datetime'].dt.month
X['day'] = X['datetime'].dt.day
X['hour'] = X['datetime'].dt.hour
X['minute'] = X['datetime'].dt.minute
X['second'] = X['datetime'].dt.second
X['dayofweek'] = X['datetime'].dt.dayofweek

X.head()
X.describe()

X.drop('minute', axis='columns', inplace=True)
X.drop('second', axis='columns', inplace=True)
X.drop('day', axis='columns', inplace=True)
X.head()

msno.matrix(df=X)

f, ax = plt.subplots(nrows=4, figsize=(10,20))

sns.barplot(data=X, y='count', x='year', ax=ax[0])
sns.barplot(data=X, y='count', x='month', ax=ax[1])
sns.barplot(data=X, y='count', x='day', ax=ax[2])
sns.barplot(data=X, y='count', x='hour', ax=ax[3])

plt.show()

f, ax = plt.subplots(nrows=4, figsize=(10, 20))

sns.pointplot(data=X, y='count', x='hour', hue='workingday',ax=ax[0])
sns.pointplot(data=X, y='count', x='hour', hue='season', ax=ax[1])
sns.pointplot(data=X, y='count', x='hour', hue='weather', ax=ax[2])
sns.pointplot(data=X, y='count', x='hour', hue='holiday', ax=ax[3])

plt.show()

plt.figure(figsize=(30, 30))
sns.heatmap(X.corr())
plt.show()

X.columns
# ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count', 'year', 'month', 'hour', 'dayofweek']
f, ax = plt.subplots(3, 2, figsize=(10, 10))

sns.boxplot(data=X, y='count', x='season', ax=ax[0, 0])
sns.boxplot(data=X, y='count', x='holiday', ax=ax[0, 1])
sns.boxplot(data=X, y='count', x='workingday', ax=ax[1, 0])
sns.boxplot(data=X, y='count', x='weather', ax=ax[1, 1])
sns.boxplot(data=X, y='count', x='dayofweek', ax=ax[2, 0])
plt.show()


f, ax = plt.subplots(ncols=3, figsize=(10, 10))
sns.scatterplot(data=X, y='count', x='temp', ax=ax[0])
sns.scatterplot(data=X, y='count', x='humidity', ax=ax[1])
sns.scatterplot(data=X, y='count', x='windspeed', ax=ax[2])
plt.show()

X['windspeed'].describe()

X.loc[X['windspeed'] == 0, 'windspeed'] = None

X.isnull().sum()

X.fillna(X['windspeed'].mean(), inplace=True)

X[X['windspeed']==0]


c_q1 = X['count'].quantile(0.25)
c_q3 = X['count'].quantile(0.75)

c_IQR = c_q3 - c_q1

c_condition = X[(X['count'] > c_q3 + c_IQR * 1.5) | (X['count'] < c_q1 - c_IQR * 1.5)]

c_condition

X.drop(c_condition.index, axis=0, inplace=True)

y = pd.DataFrame(X['count'], columns=['count'])
y

X.drop('count', axis=1, inplace=True)
X.drop('datetime', axis=1, inplace=True)
polynomial_transformer = PolynomialFeatures(4)
polynomial_features = polynomial_transformer.fit_transform(X.values)
features = polynomial_transformer.get_feature_names(X.columns)
X = pd.DataFrame(polynomial_features, columns=features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape
X_test.shape
y_train.shape
y_test.shape


model = Lasso(alpha=1, max_iter=2000, normalize=True)

model.fit(X_train, y_train)

y_train_predict = model.predict(X_train)

y_test_predict = model.predict(X_test)


mse = mean_squared_error(y_train, y_train_predict)

print("training set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')

mse = mean_squared_error(y_test, y_test_predict)

print("testing set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')

model = LinearRegression()

model.fit(X_train, y_train)

y_train_predict = model.predict(X_train)

y_test_predict = model.predict(X_test)

y_train_predict

y_test_predict

mse = mean_squared_error(y_train, y_train_predict)

print("training set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')

mse = mean_squared_error(y_test, y_test_predict)

print("testing set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')