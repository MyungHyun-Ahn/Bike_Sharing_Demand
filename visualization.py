import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import pandas as pd
import numpy as np

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

f, ax = plt.subplots(nrows=5, figsize=(10, 20))
sns.barplot(data=X, y='count', x='year', ax=ax[0])
sns.barplot(data=X, y='count', x='month', ax=ax[1])
sns.barplot(data=X, y='count', x='day', ax=ax[2])
sns.barplot(data=X, y='count', x='hour', ax=ax[3])
sns.barplot(data=X, y='count', x='dayofweek', ax=ax[4])
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

sns.scatterplot(data=X, x='temp', y='count', ax=ax[0, 0])
sns.scatterplot(data=X, x='atemp', y='count', ax=ax[0, 1])
sns.scatterplot(data=X, x='humidity', y='count', ax=ax[1, 0])
sns.scatterplot(data=X, x='windspeed', y='count', ax=ax[1, 1])

plt.show()

# windspeed에 0값이 많이 몰려있는 것을 확인 / humidity에도 0과 100 값이 몰려있는 것을 확인

w_zero_values = X[X['windspeed'] == 0]
w_zero_values

X.drop(w_zero_values.index, axis=0, inplace=True)

X['windspeed'].describe()

w_q1 = X['windspeed'].quantile(0.25)
w_q3 = X['windspeed'].quantile(0.75)

w_IQR = w_q3 - w_q1

w_outlier = (X['windspeed'] > (w_q3 + 1.5 * w_IQR)) | (X['windspeed'] < (w_q1 - 1.5 * w_IQR))

w_outlier

X[w_outlier]

remove_outlier = X.drop(X[w_outlier].index, axis=0)

remove_outlier['windspeed'].describe()

remove_outlier[w_outlier]

remove_outlier['windspeed'].describe()
# remove_outlier >> 이상치 제거한 값

ro_zero_values = remove_outlier[remove_outlier['windspeed']==0]

remove_outlier.drop(ro_zero_values.index, axis=0, inplace=True)

remove_outlier['windspeed'].describe()
(w_q1 - 1.5 * w_IQR)