import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import missingno as msno

import pandas as pd
import numpy as np

import os

from tool.scaler import *




TRAIN_FILE_PATH = 'data/train.csv'
df = pd.read_csv(TRAIN_FILE_PATH, parse_dates=['datetime'])

df.head()

df.info()

# 결측치 확인
df.isnull().sum()

# 결측치 유무 시각화
msno.matrix(df=df, figsize=(10,9))
plt.show()

df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek

f, ax = plt.subplots(nrows=5, figsize=(10,10))
sns.barplot(data=df, x='year', y='count', ax=ax[0])
sns.barplot(data=df, x='month', y='count', ax=ax[1])
sns.barplot(data=df, x='day', y='count', ax=ax[2])
sns.barplot(data=df, x='hour', y='count', ax=ax[3])
sns.barplot(data=df, x='dayofweek', y='count', ax=ax[4])
ax[4].set_xticklabels(['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'])
plt.show()

# day feature 는 19일 까지 밖에 없고 고른 분포를 보이기 때문에 삭제
df.drop('day', axis=1, inplace=True)

df.columns

f, ax = plt.subplots(nrows=3, figsize=(10,10))
sns.pointplot(data=df, y='count', x='hour', hue='season', ax=ax[0])
sns.pointplot(data=df, y='count', x='hour', hue='holiday', ax=ax[1])
sns.pointplot(data=df, y='count', x='hour', hue='workingday', ax=ax[2])
plt.show()


# 상관계수 확인을 위한 히트맵 그래프
f = plt.figure(figsize=(10,10))
sns.heatmap(data=df.corr(), annot=True)
plt.show()

# temp와 atemp / registered 와 count / season 과 month 가 높은 상관 관계를 보임
# 하지만 registered 와 count는 타겟 변수(y) 이므로 제외

from statsmodels.stats.outliers_influence import variance_inflation_factor

# 독립변수 X ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'day', 'hour', 'dayofweek']

X_col = ['season', 'temp', 'atemp', 'month']

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(df[X_col], i) for i in range(df[X_col].shape[1])]
vif['feateres'] = df[X_col].columns
vif

# vif 가 10을 넘는다면 다중공선성으로 진단 가능
# vif 가 10을 넘는 목록 ['season', 'temp', 'atemp', 'month']

# 해결방법 1. 정규화 
# 미리 만들어둔 모듈의 함수로 진행

# 노말라이즈한 데이터프레임
# season 와 month 데이터는 범주형 데이터기 때문에 정규화 하지 않음
multicol = ['temp', 'atemp']

# minmax scaling 정규화
df_n = MMscaler(df, multicol)
df_n

vif_n = pd.DataFrame()
vif_n['VIF Factor'] = [variance_inflation_factor(df_n[X_col], i) for i in range(df_n[X_col].shape[1])]
vif_n['feateres'] = df_n[X_col].columns
vif_n

'''
   VIF Factor feateres
0   86.558560   season
1  244.435192     temp
2  259.819625    atemp
3   71.657048    month
'''

'''
   VIF Factor feateres
0   86.062461   season
1  224.015624     temp
2  240.300949    atemp
3   71.498267    month
'''

sns.pairplot(df[multicol], diag_kind='hist')
plt.show()

# 정규화는 다중공선성 문제를 해결하지 못했음..

# 해결방법 2. 의존적인 변수 삭제
# temp 와 atemp / month 와 season 중 VIF 더 높은 atemp 와 month 제거

df_n.drop(['atemp', 'month'], axis=1, inplace=True)

df_n.head()

# 남아있는 독립변수 ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'humidity', 'windspeed', 'year', 'day', 'hour', 'dayofweek'] 중
# 양적 변수 ['temp', 'humidity', 'windspeed'] 와 종속변수 count 사이의 산점도 그래프

f, ax = plt.subplots(nrows=3, figsize=(8,8))
sns.scatterplot(data=df_n, x='temp', y='count',ax=ax[0])
sns.scatterplot(data=df_n, x='humidity', y='count',ax=ax[1])
sns.scatterplot(data=df_n, x='windspeed', y='count',ax=ax[2])
plt.tight_layout()
plt.show()

# 온도는 고른 분포를 보이고 있음 그러나 습도와 풍속에 0값 100값이 들어가 있음
# 이후 feature scaling 에서 처리 예정

# 먼저 범주형 데이터 One - Hot - encoding 진행 
# ['season', 'holiday', 'workingday', 'weather', 'dayofweek']

one_hot_data = ['season', 'holiday', 'workingday', 'weather', 'dayofweek']

for name in one_hot_data:
   df_n = pd.get_dummies(df_n, columns=[name])

# 타겟 변수인 casual 과 registered 제거
del_data = ['casual', 'registered']
df_n.drop(del_data, axis=1, inplace=True)

# datetime 데이터 제거
df_n.drop('datetime', axis=1, inplace=True)

df_n.columns

# 습도가 0이고 100인값 저장
hu0100 = df_n[(df_n['humidity'] == 100) | (df_n['humidity']==0)]
hu0100

hu_pre_X = hu0100.drop('humidity', axis=1)
hu_pre_X.head()

# 습도 트레이닝 데이터
hu_X = df_n.drop(hu0100.index, axis=0)
hu_y = pd.DataFrame(hu_X['humidity'], columns=['humidity'])
hu_y.head()

# 입력변수에서 타겟변수 제거
hu_X.drop('humidity', axis=1, inplace=True)

# 습도 예측
# 우선 기본 값으로 진행 후 나중에 그리드 서치 진행 예정

# k겹 교차 검증을 위해 임포트
from sklearn.model_selection import cross_val_score

# 그라디언트 부스트
from sklearn.ensemble import GradientBoostingRegressor

h_GBmodel = GradientBoostingRegressor()

h_GBmodel_scores = cross_val_score(h_GBmodel, hu_X, hu_y.values.ravel(), scoring="neg_mean_squared_error", cv=5)
h_GBmodel_scores = np.sqrt(-1 * h_GBmodel_scores)
h_GBmodel_scores.mean()  # 13.569717521587702

# 랜덤 포레스트
from sklearn.ensemble import RandomForestRegressor

h_RFmodel = RandomForestRegressor()

h_RFmodel_scores = cross_val_score(h_RFmodel, hu_X, hu_y.values.ravel(), scoring="neg_mean_squared_error", cv=5)
h_RFmodel_scores = np.sqrt(-1 * h_RFmodel_scores)
h_RFmodel_scores.mean()  # 14.025072428661328

# 배깅
from sklearn.ensemble import BaggingRegressor

h_BGmodel = BaggingRegressor()
h_BGmodel_scores = cross_val_score(h_BGmodel, hu_X, hu_y.values.ravel(), scoring="neg_mean_squared_error", cv=5)
h_BGmodel_scores = np.sqrt(-1 * h_BGmodel_scores)
h_BGmodel_scores.mean()  # 14.592442150216844

# 에다부스트
from sklearn.ensemble import AdaBoostRegressor

h_ABmodel = AdaBoostRegressor()
h_ABmodel_scores = cross_val_score(h_ABmodel, hu_X, hu_y.values.ravel(), scoring="neg_mean_squared_error", cv=5)
h_ABmodel_scores = np.sqrt(-1 * h_ABmodel_scores)
h_ABmodel_scores.mean()  # 14.833245943801334

# 그라디언드 부스트 적용
h_GBmodel.fit(hu_X, hu_y.values.ravel())
h_y_predict = h_GBmodel.predict(hu_pre_X)

h_y_predict = h_y_predict.astype('int')
h_y_predict

change_idx_list = list(hu0100['humidity'].index)
len(h_y_predict)
len(change_idx_list)

for i in range(len(change_idx_list)):
   df_n.loc[change_idx_list[i], 'humidity'] = h_y_predict[i]


# 풍속 0 데이터 예측 / 그라디언트 부스트
wind0 = df_n[df_n['windspeed']==0]
wind0

wind_pre_X = wind0.drop('windspeed', axis=1)

wind_pre_X

wind_X = df_n.drop(wind0.index, axis=0)
wind_y = pd.DataFrame(wind_X['windspeed'], columns=['windspeed'])
wind_X.drop('windspeed', axis=1, inplace=True)

w_GBmodel = GradientBoostingRegressor(n_estimators=4000, alpha=0.01)
w_GBmodel.fit(wind_X, wind_y.values.ravel())
w_y_predict = w_GBmodel.predict(wind_pre_X)
w_y_predict

wchange_idx_list = list(wind0['humidity'].index)

for i in range(len(wchange_idx_list)):
   df_n.loc[wchange_idx_list[i], 'windspeed'] = w_y_predict[i]

df_n.columns

# 연속형 데이터 ['temp', 'humidity', 'windspeed', 'count'] / 이상치 제거
Outscaler(df_n, ['temp', 'humidity', 'windspeed', 'count'])

# 타겟 변수 예측 준비

y_train = pd.DataFrame(data=df_n['count'], columns=['count'])
x_train = df_n.drop('count', axis=1)

# 테스트 준비
TEST_CSV_PATH = 'data/test.csv'
test = pd.read_csv(TEST_CSV_PATH, parse_dates=['datetime'])

test['year'] = test['datetime'].dt.year
test['hour'] = test['datetime'].dt.hour
test['dayofweek'] = test['datetime'].dt.dayofweek

test.drop('atemp', axis=1, inplace=True)
test.drop('datetime', axis=1, inplace=True)

one_hot_data = ['season', 'holiday', 'workingday', 'weather', 'dayofweek']

for name in one_hot_data:
   test = pd.get_dummies(test, columns=[name])

len(test.columns)
len(x_train.columns)



GBmodel = GradientBoostingRegressor(n_estimators=4000, alpha=0.01)
GBmodel.fit(x_train, y_train.values.ravel())
y_predict = GBmodel.predict(test)

h_GBmodel_scores = cross_val_score(GBmodel, x_train, y_train.values.ravel(), scoring="neg_mean_squared_error", cv=5)
h_GBmodel_scores = np.sqrt(-1 * h_GBmodel_scores)
h_GBmodel_scores.mean() # 75.39679599115622 기본 파라미터


SUB_FILE_PATH = 'data/sampleSubmission.csv'
sub = pd.read_csv(SUB_FILE_PATH)

sub['count'] = y_predict
sub['count'] = sub['count'].apply(lambda x:0 if x<0 else x)
sub['count']

sub.to_csv("GradientBoosting_v0.12.csv",index = False)

y_train['count'].skew()
y_train['count'].kurt()