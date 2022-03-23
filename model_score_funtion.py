import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error

from math import sqrt

def rmse(y, y_predict):
    y = np.array(y)
    y_predict = np.array(y_predict)
    rmse_score = sqrt(mean_squared_error(y, y_predict))
    return rmse_score

def rmsle(y, y_predict):
    y = np.array(y)
    y_predict = np.array(y_predict)

    log_y = np.log(y + 1)
    log_predict = np.log(y_predict + 1)

    diff = np.square(log_predict - log_y)

    diff_mean = diff.mean()

    rmsle_score = np.sqrt(diff_mean)

    return rmsle_score


def test():
    y = 90
    y_predict = 100

    rmse_score = rmse(y, y_predict)
    rmsle_score = rmsle(y, y_predict)
    print('RMSE: ', rmse_score)
    print('RMSLE: ', rmsle_score)

if __name__=='__main__':
    test()