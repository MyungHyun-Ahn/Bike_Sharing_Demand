import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

__ALL__ = ['barplots', 'hue_pointplots', 'scatterplots']


# 컬럼리스트를 받아와서 ydata에 대한 막대그래프를 연속적으로 그려주는 함수
def barplots(df, columnList, ydata):
    f, ax = plt.subplots(nrows=len(columnList), figsize=(10,10))
    for i in range(len(columnList)):
        sns.barplot(data=df, x=columnList[i], y=ydata, ax=ax[i])
    plt.tight_layout()
    plt.show()

# 포인트 플롯을 특징에 따라 나누어 그려주는 함수
def hue_pointplots(df, x, y, hueList):
    f, ax = plt.subplots(nrows=len(hueList), figsize=(10, 10))
    for i in range(len(hueList)):
        sns.pointplot(data=df, y=y, x=x, hue=hueList[i], ax=ax[i])
    plt.tight_layout()
    plt.show()

# 컬럼리스트를 받아와 y에 대한 산점도 그래프를 그려주는 함수
def scatterplots(df, columnsList, ydata):
    f, ax = plt.subplots(nrows=len(columnsList), figsize=(10, 10))
    for i in range(len(columnsList)):
        sns.scatterplot(data=df, y=ydata, x=columnsList[i], hue=columnsList[i], ax=ax[i])
    plt.tight_layout()
    plt.show()