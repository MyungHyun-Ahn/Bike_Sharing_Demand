import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import pandas as pd
import numpy as np

from math import sqrt

TRAIN_FILE_PATH = 'data/train.csv'
# datetime 컬럼을 datetime 타입으로 저장해주기 위해 parse_dates 파라미터를 사용
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

# x 라벨이 가려져서 간격 조정
plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=0.35)

sns.barplot(data=X, y='count', x='year', ax=ax[0])
sns.barplot(data=X, y='count', x='month', ax=ax[1])
sns.barplot(data=X, y='count', x='day', ax=ax[2])
sns.barplot(data=X, y='count', x='hour', ax=ax[3])
sns.barplot(data=X, y='count', x='dayofweek', ax=ax[4])

# 요일 x 라벨을 숫자보다는 보기 편하게 영문으로 바꾸어줌
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

# temp와 atemp 그리고 month 와 season 은  높은 상관 관계를 가지기 때문에 다중 공선성을 보인다고 생각하고 한개씩 삭제

X.drop(['atemp', 'month'], axis='columns', inplace=True)

# casual 과 registered 는 test 데이터에도 없고 count 예측과는 상관없는 데이터이기 때문에 삭제

X.drop(['casual', 'registered'], axis='columns', inplace=True)

# datetime 데이터는 저장타입이 datetime이기도 하고 년 월 날짜 시간으로 분류하여 저장하였기 때문에 삭제
X.drop('datetime', axis='columns', inplace=True)

# y 데이터 프레임을 만들어서 target 변수인 'count' 컬럼을 만들어 넣어줌
y = pd.DataFrame(X['count'], columns=['count'])

# X 데이터 프레임에서는 target 변수인 'count' 컬럼을 지워줌
X.drop('count', axis='columns', inplace=True)

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 간단히 모델 평가를 해보기 위한 train 셋과 test 셋 분류
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 잘 분류가 되었는지 확인
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# 회귀 모델

# 선형 회귀 모델 평균 제곱의 오차
linear_m = LinearRegression()

linear_m.fit(X_train, y_train)

linear_m_predict_y = linear_m.predict(X_test)

linear_m_mse = sqrt(mean_squared_error(y_test, linear_m_predict_y))

linear_m_mse  
# 124.33815857009867

# 선형 회귀 모델 k-fold
linear_m2 = LinearRegression()

linear_m2_scores = cross_val_score(linear_m2, X, y, scoring="neg_mean_squared_error", cv=5)
linear_m2_scores = np.sqrt(-1 * linear_m2_scores)
linear_m2_scores.mean()  
# 123.61831320404738

# 다항 회귀 모델 준비

from sklearn.preprocessing import PolynomialFeatures

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
poly_mse 
# 100.93270051766473

# 다항 회귀 모델 k-fold
poly_m2 = LinearRegression()

poly_m2_scores = cross_val_score(poly_m2, poly_X, y, scoring="neg_mean_squared_error", cv=5)
poly_m2_scores = np.sqrt(-1 * poly_m2_scores)
poly_m2_scores.mean() 
# 113.6581596110982

# L1 정규화를 위한 Lasso 모델

from sklearn.linear_model import Lasso

# Lasso 평균 제곱 오차
lasso_m = Lasso(alpha=1, max_iter=2000, normalize=True)
lasso_m.fit(poly_X_train, poly_y_train)
lasso_y_test_predict = lasso_m.predict(poly_X_test)
lasso_y_test_predict
lasso_mse = sqrt(mean_squared_error(y_test, lasso_y_test_predict))
lasso_mse 
# 157.59495896539266

# Lasso k-fold
lasso_m2 = Lasso(alpha=1, max_iter=2000, normalize=True)
lasso_m2_scores = cross_val_score(lasso_m2, poly_X, y, scoring="neg_mean_squared_error", cv=5)
lasso_m2_scores = np.sqrt(-1 * lasso_m2_scores)
lasso_m2_scores.mean() 
# 159.61076851182779

# L2 정규화를 위한 Ridge 모델

from sklearn.linear_model import Ridge

# Ridge 평균 제곱 오차
ridge_m = Ridge(alpha=0.01, max_iter=3000, normalize=True)
ridge_m.fit(poly_X_train, poly_y_train)
ridge_y_test_predict = lasso_m.predict(poly_X_test)
ridge_y_test_predict
ridge_mse = sqrt(mean_squared_error(y_test, ridge_y_test_predict))
ridge_mse 
# 157.59495896539266

# Ridge k-fold
ridge_m2 = Ridge(alpha=0.01, max_iter=2000, normalize=True)
ridge_m2_scores = cross_val_score(ridge_m2, poly_X, y, scoring="neg_mean_squared_error", cv=5)
ridge_m2_scores = np.sqrt(-1 * ridge_m2_scores)
ridge_m2_scores.mean() 
# 107.46983050400426

# 앙상블 모델

# 그라디언트 부스트 모델

from sklearn.ensemble import GradientBoostingRegressor

# GradientBoosting 평균 제곱 오차
GB_m = GradientBoostingRegressor(n_estimators=4000, alpha=0.01)
GB_m.fit(X_train, y_train.values.ravel())
GB_y_test_predict = GB_m.predict(X_test)
GB_mse = sqrt(mean_squared_error(y_test, GB_y_test_predict))
GB_mse 
# 41.4939563124132

# GradientBoosting k-fold
GB_m2 = GradientBoostingRegressor(n_estimators=4000, alpha=0.01)
GB_m2_scores = cross_val_score(GB_m2, X, y.values.ravel(), scoring="neg_mean_squared_error", cv=5)
GB_m2_scores = np.sqrt(-1 * GB_m2_scores)
GB_m2_scores.mean() 
# 62.47338661764495

# 랜덤 포레스트 모델

from sklearn.ensemble import RandomForestRegressor

# RandomForest 평균 제곱 오차
RF_m = RandomForestRegressor(n_estimators=100)
RF_m.fit(X_train, y_train.values.ravel())
RF_y_test_predict = RF_m.predict(X_test)
RF_y_test_predict
RF_mse = sqrt(mean_squared_error(y_test, RF_y_test_predict))
RF_mse 
# 43.493486694299925

# RandomForest k-fold
RF_m2 = RandomForestRegressor(n_estimators=100)
RF_m2_scores = cross_val_score(RF_m2, X, y.values.ravel(), scoring="neg_mean_squared_error", cv=5) 
RF_m2_scores = np.sqrt(-1 * RF_m2_scores)
RF_m2_scores.mean()
# 73.21367394254486

# 에다부스트

from sklearn.ensemble import AdaBoostRegressor

# AdaBoosting 평균 제곱 오차
AB_m = AdaBoostRegressor(n_estimators=100)
AB_m.fit(X_train, y_train)
AB_y_test_predict = AB_m.predict(X_test)
AB_mse = sqrt(mean_squared_error(y_test, AB_y_test_predict))
AB_mse 
# 116.34704534569923

# AdaBoosting k-fold
AB_m2 = AdaBoostRegressor(n_estimators=100)
AB_m2_scores = cross_val_score(AB_m2, X, y.values.ravel(), scoring="neg_mean_squared_error", cv=5) 
AB_m2_scores = np.sqrt(-1 * AB_m2_scores)
AB_m2_scores.mean() 
# 122.89016076704408

# 배깅 Bagging

from sklearn.ensemble import BaggingRegressor

# Bagging 평균 제곱 오차
BG_m = BaggingRegressor(n_estimators=100)
BG_m.fit(X_train, y_train)
BG_y_test_predict = BG_m.predict(X_test)
BG_mse = sqrt(mean_squared_error(y_test, BG_y_test_predict))
BG_mse 
# 45.717563391405406

# Bagging k-fold
BG_m2 = BaggingRegressor(n_estimators=100)
BG_m2_scores = cross_val_score(BG_m2, X, y.values.ravel(), scoring="neg_mean_squared_error", cv=5)
BG_m2_scores = np.sqrt(-1 * BG_m2_scores)
BG_m2_scores.mean() 
# 72.99775759783618


y.describe()

rmse_model_score_dict = {
    '선형 회귀 모델' : 124,
    '2차항 회귀 모델' : 100,
    'Lasso 모델' : 157,
    'Ridge 모델' : 157,
    'GradientBoosting 모델' :  41,
    'RandomForest 모델' : 43,
    'AdaBoosting 모델' : 116,
    'Bagging 모델' : 45
}

sorted(rmse_model_score_dict.items(), key=lambda x:x[1])
# ('GradientBoosting 모델', 41), 
# ('RandomForest 모델', 43), 
# ('Bagging 모델', 45), 
# ('2차항 회귀 모델', 100), 
# ('AdaBoosting 모델', 116), 
# ('선형 회귀 모델', 124), 
# ('Lasso 모델', 157), 
# ('Ridge 모델', 157)

k_fold_model_score_dict = {
    '선형 회귀 모델' : 123,
    '2차항 회귀 모델' : 113,
    'Lasso 모델' : 159,
    'Ridge 모델' : 107,
    'GradientBoosting 모델' :  62,
    'RandomForest 모델' : 73,
    'AdaBoosting 모델' : 122,
    'Bagging 모델' : 72
}

sorted(k_fold_model_score_dict.items(), key=lambda x:x[1])
# ('GradientBoosting 모델', 62), 
# ('Bagging 모델', 72), 
# ('RandomForest 모델', 73), 
# ('Ridge 모델', 107), 
# ('2차항 회귀 모델', 113), 
# ('AdaBoosting 모델', 122), 
# ('선형 회귀 모델', 123), 
# ('Lasso 모델', 159)

# GradientBoosting 모델이 가장 성능이 좋음

X.columns

# ['season', 'holiday', 'workingday', 'weather', 'temp', 'humidity', 'windspeed', 'year', 'hour', 'dayofweek']
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'humidity', 'windspeed', 'year', 'hour', 'dayofweek']
SUB_FILE_PATH = 'data/sampleSubmission.csv'
sub = pd.read_csv(SUB_FILE_PATH)

TEST_FILE_PATH ='data/test.csv'
test = pd.read_csv(TEST_FILE_PATH, parse_dates=['datetime'])

test.columns

test['year'] = test['datetime'].dt.year
test['hour'] = test['datetime'].dt.hour
test['dayofweek'] = test['datetime'].dt.dayofweek

test.drop(['datetime', 'atemp'], axis=1, inplace=True)



GB_m3 = GradientBoostingRegressor(n_estimators=4000, alpha=0.01)
GB_m3.fit(X, y.values.ravel())
GB_m3_y_test_predict = GB_m3.predict(test)

GB_m3_y_test_predict

sub['count'] = GB_m3_y_test_predict

sub.head()

sub['count'] = sub['count'].apply(lambda x:0 if x<0 else x)
sub['count']

sub.to_csv("GradientBoosting_v0.1.csv",index = False)

# Score 0.77242 2553등 쯤 3243명 중

