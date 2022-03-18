import imp
from re import X
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

x_train = pd.read_csv('data/train.csv', parse_dates=['datetime'])
y_train = pd.read_csv('data/test.csv', parse_dates=['datetime'])

null = x_train.isnull().sum()
print(null)

plt.style.use('seaborn')

# matrix로 결측치 확인
msno.matrix(df=x_train)

# ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']

x_train['year'] = x_train['datetime'].dt.year
x_train['month'] = x_train['datetime'].dt.month
x_train['day'] = x_train['datetime'].dt.day
x_train['hour'] = x_train['datetime'].dt.hour
x_train['minute'] = x_train['datetime'].dt.minute
x_train['second'] = x_train['datetime'].dt.second

y_train['year'] = y_train['datetime'].dt.year
y_train['month'] = y_train['datetime'].dt.month
y_train['day'] = y_train['datetime'].dt.day
y_train['hour'] = y_train['datetime'].dt.hour
y_train['minute'] = y_train['datetime'].dt.minute
y_train['second'] = y_train['datetime'].dt.second

#print(x_train.head())

f1, ax1 = plt.subplots(2, 2, figsize=(15,10))

sns.barplot(data=x_train, y='count', x='year', ax=ax1[0, 0])
ax1[0, 0].set(ylabel='count', title='Rental amount by year')

sns.barplot(data=x_train, y='count', x='month', ax=ax1[0, 1])
ax1[0, 1].set(ylabel='count', title='Rental amount by month')

sns.barplot(data=x_train, y='count', x='day', ax=ax1[1, 0])
ax1[1, 0].set(ylabel='count', title='Rental amount by day')

sns.barplot(data=x_train, y='count', x='hour', ax=ax1[1, 1])
ax1[1, 1].set(ylabel='count', title='Rental amount by hour')


# workingday holiday season weather

f2, ax2 = plt.subplots(nrows=5, figsize=(20, 20))

sns.pointplot(data=x_train, y='count', x='hour', ax=ax2[0])
sns.pointplot(data=x_train, y='count', x='hour', hue='workingday', ax=ax2[1])
sns.pointplot(data=x_train, y='count', x='hour', hue='holiday', ax=ax2[2])
sns.pointplot(data=x_train, y='count', x='hour', hue='season', ax=ax2[3])
sns.pointplot(data=x_train, y='count', x='hour', hue='weather', ax=ax2[4])

corr_data = x_train[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']]

f3, ax3 = plt.subplots(figsize=(20, 20))

sns.heatmap(data=corr_data.corr(), annot=True)

f3, ax3 = plt.subplots(ncols=3, figsize=(10, 10))

sns.scatterplot(data=x_train, y='count', x='temp', ax=ax3[0])
sns.scatterplot(data=x_train, y='count', x='humidity', ax=ax3[1])
sns.scatterplot(data=x_train, y='count', x='windspeed', ax=ax3[2])

f4, ax4 = plt.subplots(ncols=3, figsize=(10, 10))

sns.boxplot(data=x_train, y='temp', ax=ax4[0])
sns.boxplot(data=x_train, y='humidity', ax=ax4[1])
sns.boxplot(data=x_train, y='windspeed', ax=ax4[2])

# boxplot 확인 결과 temp에는 이상치가 없고 humidity와 windspeed에는 있는 것 같음
# 이상치 제거
print(x_train['humidity'].describe())
print(x_train['windspeed'].describe())

h_q1 = x_train['humidity'].quantile(0.25)
h_q3 = x_train['humidity'].quantile(0.75)

h_IQR = h_q3 - h_q1

h_condition = x_train[(x_train['humidity'] > h_q3 + h_IQR * 1.5) | (x_train['humidity'] < h_q1 - h_IQR * 1.5)]

print(h_condition)

x_train.drop(h_condition.index, inplace=True)

w_q1 = x_train['windspeed'].quantile(0.25)
w_q3 = x_train['windspeed'].quantile(0.75)

w_IQR = w_q3 - w_q1

w_condition = x_train[(x_train['windspeed'] > w_q3 + w_IQR * 1.5) | (x_train['windspeed'] < w_q1 - w_IQR * 1.5)]

print(w_condition)

x_train.drop(w_condition.index, inplace=True)

# boxplot 그래프를 그려 이상치가 제거 되었는지 확인

f5, ax5 = plt.subplots(ncols=2, figsize=(10,10))

sns.boxplot(data=x_train, y='humidity', ax=ax5[0])
sns.boxplot(data=x_train, y='windspeed', ax=ax5[1])

#plt.show()

# 이상치가 제거 되었음
# 회원 비회원은 예측에 관계가 없기 때문에 삭제

x_train.drop(['registered', 'casual'], axis=1, inplace=True)


# 선형 회귀 모델
# ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count', 'year', 'month', 'day', 'hour', 'minute', 'second']

main_train = x_train.drop('count', axis=1)

print(main_train.head())

main_target = pd.DataFrame(data=x_train['count'], columns=['count'])

print(main_target.head())


print(main_train.head())
print(y_train.head())

print(main_train.shape)
print(main_target.shape)
print(y_train.shape)

main_train.drop('datetime', axis=1, inplace=True)
y_train = y_train.drop('datetime', axis=1)

a_train, a_test, b_train, b_test = train_test_split(main_train, main_target, test_size=0.2)

a_train.shape

a_test.shape

b_train.shape

b_test.shape


linear_model = LinearRegression()
linear_model.fit(main_train, main_target)

main_train
y_target_predict = linear_model.predict(y_train)

print(y_target_predict)

y_train
y_target_predict

main_train.columns


linear_model2 = LinearRegression()

linear_model2.fit(a_train, b_train)

b_target_predict = linear_model2.predict(a_test)

linear_model2.coef_

linear_model2.intercept_

mse = mean_squared_error(b_test, b_target_predict) ** 0.5

mse

b_target_predict