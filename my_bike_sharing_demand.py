import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import pandas as pd
import numpy as np

from math import sqrt
TRAIN_FILE_PATH = 'data/train.csv'
X = pd.read_csv(TRAIN_FILE_PATH, parse_dates=['datetime'])

X.info()

msno.matrix(df=X)
plt.show()
# 결측치 확인 없음.

# 데이터 확인 결과 분과 초는 모두 0이기 때문에 분류하지 않음

X['year'] = X['datetime'].dt.year
X['month'] = X['datetime'].dt.month
X['day'] = X['datetime'].dt.day
X['hour'] = X['datetime'].dt.hour

# 요일 별 정보도 궁금하기 때문에 분류
X['dayofweek'] = X['datetime'].dt.dayofweek

f, ax = plt.subplots(nrows=5, figsize=(6, 12))
plt.suptitle('Time Data Visualization')
plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=0.35)
sns.barplot(data=X, y='count', x='year', ax=ax[0])
sns.barplot(data=X, y='count', x='month', ax=ax[1])
sns.barplot(data=X, y='count', x='day', ax=ax[2])
sns.barplot(data=X, y='count', x='hour', ax=ax[3])
sns.barplot(data=X, y='count', x='dayofweek', ax=ax[4])
ax[4].set_xticklabels(['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'])
plt.show()

# 시간 정보 시각화 결과 : day는 거의 균일하고 19일 까지 밖에 없기 때문에 분석에 관련이 없어보임 >> 삭제

X.drop('day', axis=1, inplace=True)

# 기상정보 : temp, atemp, humidity, windspeed

f, ax = plt.subplots(2, 2, figsize=(10, 10))

sns.histplot(data=X, x='temp', ax=ax[0, 0], bins=50)
sns.histplot(data=X, x='atemp', ax=ax[0, 1], bins=50)
sns.histplot(data=X, x='humidity', ax=ax[1, 0], bins=50)
sns.histplot(data=X, x='windspeed', ax=ax[1, 1], bins=50)

plt.show()

# windspeed 에 0인 데이터가 많은게 보임
# 산점도 그래프 그리기

f, ax = plt.subplots(2, 2, figsize=(10, 10))
plt.suptitle('Quantitative Variable Visualization')
sns.scatterplot(data=X, x='temp', y='count', color='r', ax=ax[0, 0])
sns.scatterplot(data=X, x='atemp', y='count', color='g', ax=ax[0, 1])
sns.scatterplot(data=X, x='humidity', y='count', color='b', ax=ax[1, 0])
sns.scatterplot(data=X, x='windspeed', y='count', color='y', ax=ax[1, 1])

plt.show()

# windspeed에 0값이 많이 몰려있는 것을 확인 / humidity에도 0과 100 값이 몰려있는 것을 확인
# 습도의 0과 100값을 삭제해야할까?

# 지금은 일단 풍속의 0값을 삭제하고 추후에 예측 모델을 이용하여 0값도 예측하여 채워넣을 예정
w_zero_values = X[X['windspeed'] == 0]
w_zero_values

X.drop(w_zero_values.index, axis=0, inplace=True)

# 숫자형 변수인 풍속 습도 온도 체감온도 대여횟수를 리스트에 넣어주고 반복문으로 이상치 제거
feature_list = ['windspeed', 'humidity', 'temp', 'atemp', 'count']

for name in feature_list:
    q1 = X[name].quantile(0.25)
    q3 = X[name].quantile(0.75)
    IQR = q3 - q1

    outlier = (X[name] > (q3 + 1.5 * IQR)) | (X[name] < (q1 - 1.5 * IQR))

    X = X.drop(X[outlier].index, axis=0)

X.describe()

X.columns

# ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count', 'year', 'month', 'day', 'hour', 'dayofweek']
# windspeed, humidity 이상치 제거 완료 temp와 atemp  

X = X[['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'year', 'month', 'hour', 'dayofweek', 'count']]

# count와 관련된 상관계수를 더 잘 보기 위해서 컬럼 순서 변경

plt.figure(figsize=(20,30))
sns.heatmap(X.corr(), annot=True)
plt.show()

# temp와 atemp 그리고 month 와 season 은 다중 공선성

X.drop(['atemp', 'month'], axis='columns', inplace=True)

# casual 과 registered 는 test 데이터에도 없고 count 예측과는 상관없는 데이터이기 때문에 삭제

X.drop(['casual', 'registered'], axis='columns', inplace=True)

# datetime 데이터는 년 월 날짜 시간으로 분류하여 저장하였기 때문에 삭제
X.drop('datetime', axis='columns', inplace=True)

y = pd.DataFrame(X['count'], columns=['count'])
X.drop('count', axis='columns', inplace=True)

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape
X_test.shape
y_train.shape
y_test.shape
# 선형 회귀 모델 평균 제곱의 오차
linear_m = LinearRegression()

linear_m.fit(X_train, y_train)

linear_m_predict_y = linear_m.predict(X_test)

linear_m_mse = sqrt(mean_squared_error(y_test, linear_m_predict_y))

linear_m_mse  # 124.33815857009867
# 선형 회귀 모델 k-fold
linear_m2 = LinearRegression()

linear_m2_scores = cross_val_score(linear_m2, X, y, scoring="neg_mean_squared_error", cv=5)
linear_m2_scores = np.sqrt(-1 * linear_m2_scores)
linear_m2_scores.mean()  # 123.61831320404738

from sklearn.preprocessing import PolynomialFeatures
# 다항 회귀 모델 준비
poly_trans = PolynomialFeatures(2)

poly_data = poly_trans.fit_transform(X)
poly_feature_names = poly_trans.get_feature_names(X.columns)

poly_X = pd.DataFrame(poly_data, columns=poly_feature_names)

poly_X_train, poly_X_test, poly_y_train, poly_y_test = train_test_split(poly_X, y, test_size=0.2)

# 다항 회귀 모델 평균 제곱의 오차
poly_m = LinearRegression()
poly_m.fit(poly_X_train,poly_y_train)
poly_y_test_predict = poly_m.predict(poly_X_test)
poly_y_test_predict
poly_mse = sqrt(mean_squared_error(poly_y_test, poly_y_test_predict))
poly_mse # 100.93270051766473

# 다항 회귀 모델 k-fold
poly_m2 = LinearRegression()

poly_m2_scores = cross_val_score(poly_m2, poly_X, y, scoring="neg_mean_squared_error", cv=5)
poly_m2_scores = np.sqrt(-1 * poly_m2_scores)
poly_m2_scores.mean() # 113.6581596110982

from sklearn.linear_model import Lasso

lasso_m = Lasso(alpha=1, max_iter=2000, normalize=True)
lasso_m.fit(poly_X_train, poly_y_train)
lasso_y_test_predict = lasso_m.predict(poly_X_test)
lasso_y_test_predict
lasso_mse = sqrt(mean_squared_error(y_test, lasso_y_test_predict))
lasso_mse # 157.59495896539266

lasso_m2 = Lasso(alpha=1, max_iter=2000, normalize=True)
lasso_m2_scores = cross_val_score(lasso_m2, poly_X, y, scoring="neg_mean_squared_error", cv=5)
lasso_m2_scores = np.sqrt(-1 * lasso_m2_scores)
lasso_m2_scores.mean() # 159.61076851182779

from sklearn.linear_model import Ridge

ridge_m = Ridge(alpha=0.01, max_iter=3000, normalize=True)
ridge_m.fit(poly_X_train, poly_y_train)
ridge_y_test_predict = lasso_m.predict(poly_X_test)
ridge_y_test_predict
ridge_mse = sqrt(mean_squared_error(y_test, ridge_y_test_predict))
ridge_mse # 157.59495896539266

ridge_m2 = Ridge(alpha=0.01, max_iter=2000, normalize=True)
ridge_m2_scores = cross_val_score(ridge_m2, poly_X, y, scoring="neg_mean_squared_error", cv=5)
ridge_m2_scores = np.sqrt(-1 * ridge_m2_scores)
ridge_m2_scores.mean() # 107.46983050400426