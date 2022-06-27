from pickle import dump

import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("./house_data.csv")
data_intrested_in = df[["floors", "price"]]

X = data_intrested_in.iloc[:, :1]
y = data_intrested_in.iloc[:, 1:2]

slr = LinearRegression()

slr.fit(X.values, y.values)

dump(slr, open("trained_slr_model.pkl", "wb"))
