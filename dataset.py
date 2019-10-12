import pandas as pd
import numpy as np

test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')

data = train_data.drop(['label'],axis=1)
data = data.append(test_data,ignore_index = False)
data.to_csv('data.csv',index=False)
