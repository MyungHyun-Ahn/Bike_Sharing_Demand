import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import missingno as msno

import pandas as pd
import numpy as np

import os

from tool.scaler import *
from tool.graph import *
import warnings

warnings.filterwarnings(action='ignore')


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

# barplots 함수를 graph 모듈에 만들어 사용
barplots(df, ['year', 'month', 'day', 'hour', 'dayofweek'], 'count')

'''
f, ax = plt.subplots(nrows=5, figsize=(10,10))
sns.barplot(data=df, x='year', y='count', ax=ax[0])
sns.barplot(data=df, x='month', y='count', ax=ax[1])
sns.barplot(data=df, x='day', y='count', ax=ax[2])
sns.barplot(data=df, x='hour', y='count', ax=ax[3])
sns.barplot(data=df, x='dayofweek', y='count', ax=ax[4])
ax[4].set_xticklabels(['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'])
plt.show()
'''
# day feature 는 19일 까지 밖에 없고 고른 분포를 보이기 때문에 삭제
df.drop('day', axis=1, inplace=True)

df.columns

# hue_pointplots 함수를 graph 모듈에 만들어 사용
hue_pointplots(df, 'hour', 'count', ['season', 'holiday', 'workingday'])

'''
f, ax = plt.subplots(nrows=3, figsize=(10,10))
sns.pointplot(data=df, y='count', x='hour', hue='season', ax=ax[0])
sns.pointplot(data=df, y='count', x='hour', hue='holiday', ax=ax[1])
sns.pointplot(data=df, y='count', x='hour', hue='workingday', ax=ax[2])
plt.show()
'''

# 상관계수 확인을 위한 히트맵 그래프
f = plt.figure(figsize=(10,10))
sns.heatmap(data=df.corr(), annot=True)
plt.tight_layout()
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
   정규화 수행 전
   VIF Factor feateres
0   86.558560   season
1  244.435192     temp
2  259.819625    atemp
3   71.657048    month
'''

'''
   정규화 수행 후
   VIF Factor feateres
0   86.062461   season
1  224.015624     temp
2  240.300949    atemp
3   71.498267    month
'''

sns.pairplot(df[['temp', 'atemp', 'season', 'month']], diag_kind='hist')
plt.show()

# 정규화는 다중공선성 문제를 해결하지 못했음..

# 해결방법 2. 의존적인 변수 삭제
# temp 와 atemp / month 와 season 중 VIF 더 높은 atemp 와 month 제거

df_n.drop(['atemp', 'month'], axis=1, inplace=True)

df_n.head()

# 남아있는 독립변수 ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'humidity', 'windspeed', 'year', 'day', 'hour', 'dayofweek'] 중
# 양적 변수 ['temp', 'humidity', 'windspeed'] 와 종속변수 count 사이의 산점도 그래프

scatterplots(df_n, ['temp', 'humidity', 'windspeed'], 'count')
'''
f, ax = plt.subplots(nrows=3, figsize=(8,8))
sns.scatterplot(data=df_n, x='temp', y='count',ax=ax[0])
sns.scatterplot(data=df_n, x='humidity', y='count',ax=ax[1])
sns.scatterplot(data=df_n, x='windspeed', y='count',ax=ax[2])
plt.tight_layout()
plt.show()
'''

# 온도는 고른 분포를 보이고 있음 그러나 습도와 풍속에 0값 100값이 들어가 있음
# 나중에 처리 예정

# 먼저 범주형 데이터 One - Hot - encoding 진행 
# ['season', 'holiday', 'workingday', 'weather', 'dayofweek']

one_hot_data = ['season', 'holiday', 'workingday', 'weather', 'dayofweek']

for name in one_hot_data:
   df_n = pd.get_dummies(df_n, columns=[name])


# 분포 그래프 그리기
f, ax = plt.subplots(nrows=3, figsize=(10,8))
sns.distplot(df_n['count'], ax=ax[0])
sns.distplot(df_n['casual'], ax=ax[1])
sns.distplot(df_n['registered'], ax=ax[2])
plt.tight_layout()
plt.show()

print('왜도: {}, 첨도: {}'.format(df_n['count'].skew(), df_n['count'].kurt()))  # 왜도: 1.2420662117180776, 첨도: 1.3000929518398334
print('왜도: {}, 첨도: {}'.format(df_n['casual'].skew(), df_n['casual'].kurt()))  # 왜도: 2.4957483979812567, 첨도: 7.551629305632764
print('왜도: {}, 첨도: {}'.format(df_n['registered'].skew(), df_n['registered'].kurt()))  # 왜도: 1.5248045868182296, 첨도: 2.6260809999210672

# count 만 봤을 때는 첨도가 많이 심하지 않아 보였지만
# caual과 registered 로 나누워 봤을 때는 첨도가 많이 뾰족하고
# 세 타겟 변수 모두 오른쪽으로 기울어져있고 casual이 많이 기울어져 있음

# 로그 스케일링 numpy의 log1p 함수를 사용 : 1을 더한 값에 로그를 취하는 함수

df_n['log_count'] = np.log1p(df_n['count'])
df_n['log_casual'] = np.log1p(df_n['casual'])
df_n['log_registered'] = np.log1p(df_n['registered'])

# 로그 취한 타겟 변수의 정규 분포

f, ax = plt.subplots(nrows=3, figsize=(10,8))
sns.distplot(df_n['log_count'], ax=ax[0])
sns.distplot(df_n['log_casual'], ax=ax[1])
sns.distplot(df_n['log_registered'], ax=ax[2])
plt.tight_layout()
plt.show()

# 왜도 첨도가 많이 개선 되었음
print('왜도: {}, 첨도: {}'.format(df_n['log_count'].skew(), df_n['log_count'].kurt()))  # 왜도: -0.8514116321738531, 첨도: -0.11948257101776338
print('왜도: {}, 첨도: {}'.format(df_n['log_casual'].skew(), df_n['log_casual'].kurt()))  # 왜도: -0.22472252892408062, 첨도: -0.8696974993823288
print('왜도: {}, 첨도: {}'.format(df_n['log_registered'].skew(), df_n['log_registered'].kurt()))  # 왜도: -0.8555617565369439, 첨도: -0.06995275735124729

# 정규분포를 띄는지 시각화 비교
# 조금 더 정규분포에 가까워 졌음

from scipy import stats

f = plt.figure(figsize=(10, 10))
ax1 = f.add_subplot(3, 2, 1)
res = stats.probplot(df_n['count'], plot=plt)
ax1.set_title('count probability')

ax2 = f.add_subplot(3, 2, 2)
res = stats.probplot(df_n['log_count'], plot=plt)
ax2.set_title('log_count probability')

ax3 = f.add_subplot(3, 2, 3)
res = stats.probplot(df_n['casual'], plot=plt)
ax3.set_title('casual probability')

ax4 = f.add_subplot(3, 2, 4)
res = stats.probplot(df_n['log_casual'], plot=plt)
ax4.set_title('log_casual probability')

ax5 = f.add_subplot(3, 2, 5)
res = stats.probplot(df_n['registered'], plot=plt)
ax5.set_title('registered probability')

ax6 = f.add_subplot(3, 2, 6)
res = stats.probplot(df_n['log_registered'], plot=plt)
ax6.set_title('log_registered probability')

plt.tight_layout()
plt.show()

# 나머지 연속형 데이터의 첨도와 왜도를 확인

df_n.columns
print('왜도: {}, 첨도: {}'.format(df_n['temp'].skew(), df_n['temp'].kurt()))  # 왜도: 0.003690844422470916, 첨도: -0.9145302637630794
print('왜도: {}, 첨도: {}'.format(df_n['humidity'].skew(), df_n['humidity'].kurt()))  # 왜도: -0.08633518364548581, 첨도: -0.7598175375208864
print('왜도: {}, 첨도: {}'.format(df_n['windspeed'].skew(), df_n['windspeed'].kurt()))  # 왜도: 0.5887665265853944, 첨도: 0.6301328693364932

f = plt.figure(figsize=(10, 10))
ax1 = f.add_subplot(3, 1, 1)
res = stats.probplot(df_n['temp'], plot=plt)
ax1.set_title('temp probability')

ax2 = f.add_subplot(3, 1, 2)
res = stats.probplot(df_n['humidity'], plot=plt)
ax1.set_title('humidity probability')

ax3 = f.add_subplot(3, 1, 3)
res = stats.probplot(df_n['windspeed'], plot=plt)
ax1.set_title('windspeed probability')

plt.tight_layout()
plt.show()


# 습도 데이터에 의문점이 생겨 데이터를 찾던중 0값이 연속적이지 못함을 발견
# 데이터 엑셀 파일을 살펴본 결과 100값은 자연스러움 >> 연속적
# 하루치 데이터가 빠져있음

df_n[df_n['humidity'] == 0] # 22개 > 하루치 데이터

df_n.columns

df_n[df_n['windspeed']==0] # 1313개 > 무수히 많음

# 풍속 먼저 예측

# 풍속 예측 데이터 준비
# 풍속은 날씨 관련 데이터 이므로 날씨와 관련된 데이터를 독립변수 X로 설정
# ['temp', 'humidity', 'hour', 'season_1', 'season_2', 'season_3', 'season_4', 'weather_1', 'weather_2', 'weather_3', 'weather_4']
# 다만 습도에 0값 데이터는 신뢰도가 떨어지므로 삭제


# 훈련 데이터 준비
wind_x = ['temp', 'windspeed', 'humidity', 'hour', 'season_1', 'season_2', 'season_3', 'season_4', 'weather_1', 'weather_2', 'weather_3', 'weather_4']

wind_df = df_n[wind_x]
del_hu = wind_df[wind_df['humidity'] == 0]
wind_df.drop(del_hu.index, axis=0, inplace=True)
wind_test = wind_df[wind_df['windspeed'] == 0]
wind_df.drop(wind_test.index, axis=0, inplace=True)
wind_dfX = wind_df.drop('windspeed', axis=1)
wind_dfy = pd.DataFrame(wind_df['windspeed'], columns=['windspeed'])
wind_testX = wind_test.drop('windspeed', axis=1)

# 그리드 서치를 위한 임포트
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor
# 그라디언트 부스트 파라미터
GBR_param = [{
   'n_estimators': [10, 100, 1000, 2000],
   'max_features': range(1,4),
   'max_depth': range(3,5),
   'learning_rate': np.linspace(0.1, 1, 10)
}]

GBR_model = GradientBoostingRegressor()

GBR_gcv = GridSearchCV(GBR_model, param_grid=GBR_param, scoring='r2', cv=5, n_jobs=-1)
GBR_gcv.fit(wind_dfX, wind_dfy.values.ravel())

print('베스트 하이퍼 파라미터: {0}'.format(GBR_gcv.best_params_))
print('베스트 하이퍼 파라미터 일 때 정확도: {0:.2f}'.format(GBR_gcv.best_score_))
# 베스트 하이퍼 파라미터: {'learning_rate': 0.1, 'max_depth': 3, 'max_features': 2, 'n_estimators': 100}
# 베스트 하이퍼 파라미터 일 때 정확도: 0.12

best_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, max_features=2, n_estimators=100)
best_model.fit(wind_dfX, wind_dfy)

wind_pre_y = best_model.predict(wind_testX)

wind_pre_y

len(wind_pre_y)
len(df_n[df_n['windspeed']==0])

wind_zeros = df_n[df_n['windspeed']==0]

wzeros_index = list(wind_zeros.index)

for i in range(len(wind_pre_y)):
   df_n.loc[wzeros_index[i], 'windspeed'] = wind_pre_y[i]

# 풍속 0 값 처리 완료
df_n.head(10)

df['windspeed'].head(10)

# windspeed 데이터가 연속적으로 채워짐

'''
             datetime      temp  humidity  windspeed  casual  registered  count  year  ...  dayofweek_2  dayofweek_3  dayofweek_4  dayofweek_5  dayofweek_6  log_count  log_casual  log_registered
0 2011-01-01 00:00:00  0.224490        81  10.517576       3          13     16  2011  ...            0            0            0            1            0   2.833213    1.386294        2.639057
1 2011-01-01 01:00:00  0.204082        80  10.698296       8          32     40  2011  ...            0            0            0            1            0   3.713572    2.197225        3.496508
2 2011-01-01 02:00:00  0.204082        80  10.823663       5          27     32  2011  ...            0            0            0            1            0   3.496508    1.791759        3.332205
3 2011-01-01 03:00:00  0.224490        75  11.855815       3          10     13  2011  ...            0            0            0            1            0   2.639057    1.386294        2.397895
4 2011-01-01 04:00:00  0.224490        75  11.855815       0           1      1  2011  ...            0            0            0            1            0   0.693147    0.000000        0.693147
5 2011-01-01 05:00:00  0.224490        75   6.003200       0           1      1  2011  ...            0            0            0            1            0   0.693147    0.000000        0.693147
6 2011-01-01 06:00:00  0.204082        80  11.207899       2           0      2  2011  ...            0            0            0            1            0   1.098612    1.098612        0.000000
7 2011-01-01 07:00:00  0.183673        86  11.253412       1           2      3  2011  ...            0            0            0            1            0   1.386294    0.693147        1.098612
8 2011-01-01 08:00:00  0.224490        75  12.903327       1           7      8  2011  ...            0            0            0            1            0   2.197225    0.693147        2.079442
9 2011-01-01 09:00:00  0.306122        76  14.309837       8           6     14  2011  ...            0            0            0            1            0   2.708050    2.197225        1.945910
'''

df_n['log_windspeed'] = np.log1p(df_n['windspeed'])

f = plt.figure(figsize=(10, 10))
ax1 = f.add_subplot(3, 1, 1)
res = stats.probplot(df['windspeed'], plot=plt)
ax1.set_title('Windspeed before prediction')

ax2 = f.add_subplot(3, 1, 2)
res = stats.probplot(df_n['windspeed'], plot=plt)
ax2.set_title('Windspeed after prediction')

ax3 = f.add_subplot(3, 1, 3)
res = stats.probplot(df_n['log_windspeed'], plot=plt)
ax3.set_title('After log scaling')

plt.tight_layout()
plt.show()

df_n.to_csv('windspeed_predict.csv', index=False)