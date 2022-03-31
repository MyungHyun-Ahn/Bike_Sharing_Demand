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

# 이상치 확인을 위한 연속형 데이터 박스플롯

f, ax = plt.subplots(2, 2, figsize=(8,6))
sns.boxplot(data=df_n, y='temp', ax=ax[0, 0])
sns.boxplot(data=df_n, y='humidity', ax=ax[0, 1])
sns.boxplot(data=df_n, y='windspeed', ax=ax[1, 0])
sns.boxplot(data=df_n, y='count', ax=ax[1, 1])
plt.show()

# 'humidity', 'windspeed', 'count' 에 이상치가 있음
# 이상치 제거
out_df = Outscaler(df_n, ['humidity', 'windspeed', 'count'])

# 그래프 확인
f, ax = plt.subplots(2, 2, figsize=(8,6))
sns.boxplot(data=out_df, y='temp', ax=ax[0, 0])
sns.boxplot(data=out_df, y='humidity', ax=ax[0, 1])
sns.boxplot(data=out_df, y='windspeed', ax=ax[1, 0])
sns.boxplot(data=out_df, y='count', ax=ax[1, 1])
plt.tight_layout()
plt.show()
