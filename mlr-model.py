from pickle import dump

import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("./house_data.csv")
data_intrested_in = df[["price", "bedrooms",
                        "bathrooms", "floors", "condition"]]


X = data_intrested_in.iloc[:, 1:]
y = data_intrested_in["price"]

mlr = LinearRegression()

mlr.fit(X.values, y.values)

dump(mlr, open("trained_mlr_model.pkl", "wb"))
