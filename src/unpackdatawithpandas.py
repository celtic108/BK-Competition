import pandas as pd
import numpy as np

df = pd.read_csv('../Data/train_numeric.csv', header = 0, nrows=9999)
#df = pd.read_csv('../Data/train_numeric.csv', header = 0, skiprows=10000, nrows=9999)
print type(df)
#df.types
print df.info()
print df.describe()
print df.head(3)