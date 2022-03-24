from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

__ALL__ = ['Outscaler']

# 원하는 컬럼명 리스트를 받아와 Outlier Scaling
# 이상치를 제거해주는 함수
def Outscaler(df, columnsList):
    for name in columnsList:
        q1 = df[name].quantile(0.25)
        q3 = df[name].quantile(0.75)

        IQR = q3 - q1

        outlier = (df[name] > (q3 + 1.5 * IQR)) | (df[name] < (q1 - 1.5 * IQR))

        df.drop(df[outlier].index, axis=0, inplace=True)

# Standard Scaler
# (Xi - (X평균)) / (X의 표준편차)

# 테스트 데이터 프레임을 만들고 테스트

def test():
    a = []
    b = []
    for i in range(100):
        a_n = np.random.randint(100)
        a.append(int(a_n))
        b_n = np.random.randint(100)
        b.append(int(b_n))

    a.append(999999)
    b.append(999999)
    
    a_df = pd.DataFrame(data=a, columns=['number1'])
    a_df['number2'] = b

    column_name = a_df.columns
    column_name
    a_df.describe()
    Outscaler(a_df, column_name)

    a_df.describe()

if __name__ == '__main__':
    test()




