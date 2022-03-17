import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib as mpl
import matplotlib.pyplot as plt

x_train = pd.read_csv('data/train.csv', parse_dates=['datetime'])
y_train = pd.read_csv('data/train.csv', parse_dates=['datetime'])

null = x_train.isnull().sum()
print(null)


plt.style.use('seaborn')
# ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']

x_train['year'] = x_train['datetime'].dt.year
x_train['month'] = x_train['datetime'].dt.month
x_train['day'] = x_train['datetime'].dt.day
x_train['hour'] = x_train['datetime'].dt.hour
x_train['minute'] = x_train['datetime'].dt.minute
x_train['second'] = x_train['datetime'].dt.second

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

# 온도 습도 풍속에 0 값이 있음



plt.show()
