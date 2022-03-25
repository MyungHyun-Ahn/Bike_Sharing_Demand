'''  sklearn.preprocessing에 있는 MinMaxScaler와 StandardScaler는 numpy 배열을 반환하기 때문에 데이터 프레임으로 반환하고 싶어서 제작  '''

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pandas as pd
import numpy as np

__ALL__ = ['Outscaler', 'MMscaler', 'StdScaler']

# 원하는 컬럼명 리스트를 받아와 Outlier Scaling
# 이상치를 제거해주는 함수
def Outscaler(df, columnsList):
    for name in columnsList:
        q1 = df[name].quantile(0.25)
        q3 = df[name].quantile(0.75)

        IQR = q3 - q1

        outlier = (df[name] > (q3 + 1.5 * IQR)) | (df[name] < (q1 - 1.5 * IQR))

        df.drop(df[outlier].index, axis=0, inplace=True)

# MinMax Scaler
# x = (x - x(min))/(x(max)-x(min))

def MMscaler(df, columnsList):
    mms_df = df.copy()
    for name in columnsList:
        max = df[name].max()
        min = df[name].min()
        for i in range(df.shape[0]):
            mms_df.loc[i, name] = (df.loc[i, name] - min) / (max - min)

    return mms_df


# Standard Scaler  
# (Xi - (X평균)) / (X의 표준편차)

def Stdscaler(df, columnsList):
    std_df = df.copy()
    for name in columnsList:
        mean = df[name].mean()
        std = df[name].std()
        for i in range(df.shape[0]):
            std_scale = (df.loc[i, name] - mean) / std
            std_df.loc[i, name] = std_scale
    return std_df



# 테스트 데이터 프레임을 만들고 테스트

def test():
    a = []
    b = []
    for i in range(100):
        a_n = np.random.randint(100)
        a.append(int(a_n))
        b_n = np.random.randint(100)
        b.append(int(b_n))

    # 이상치 제거 테스트를 위한 엄청나게 큰 값 추가
    a.append(999999)
    b.append(999999)
    
    a_df = pd.DataFrame(data=a, columns=['number1'])
    a_df['number2'] = b
    a_df.shape[0]
    a_df.info()
    
    column_name = a_df.columns

    out_test_df = a_df.copy()

    my_mms_test_df = a_df.copy()
    sk_mms_test_df = a_df.copy()

    my_std_test_df = a_df.copy()
    sk_std_test_df = a_df.copy()

    Outscaler(out_test_df, column_name)
    MMscaler(my_mms_test_df, column_name)
    Stdscaler(my_std_test_df, column_name)

    mms = MinMaxScaler()
    mms.fit(sk_mms_test_df)
    sk_mms_test_df = mms.transform(sk_mms_test_df)
    sk_mms_test_df = pd.DataFrame(sk_mms_test_df, columns=column_name)

    std = StandardScaler()
    std.fit(sk_std_test_df)
    sk_std_test_df = std.transform(sk_std_test_df)
    sk_std_test_df = pd.DataFrame(sk_std_test_df, columns=column_name)
    
    a_df.describe()

    out_test_df.describe()

    my_mms_test_df.describe()
    sk_mms_test_df.describe()

    my_std_test_df.describe()
    sk_std_test_df.describe()



if __name__ == '__main__':
    test()




