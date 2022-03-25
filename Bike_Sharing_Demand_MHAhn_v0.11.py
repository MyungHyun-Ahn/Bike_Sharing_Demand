import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import missingno as msno

import pandas as pd
import numpy as np

import os

import tool




TRAIN_FILE_PATH = 'data/train.csv'

df = pd.read_csv(TRAIN_FILE_PATH)

df.head()

df = MMscaler(df, ['humidity'])




df.head()
df.drop()