import pandas as pd

df=pd.read_csv("new_train_data_s140.csv",
               encoding='latin-1',
               nrows=5)
print(df.head())
print(df.columns)