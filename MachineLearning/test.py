import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# replace every ? with np.nan in dataset
raw_data = pd.read_csv("../data/raw/mixed_data1.data", sep=',', header=None)
raw_data = raw_data.replace('?', np.nan)

# splitting classes indicators from rest of the data
X = raw_data.iloc[:, :-1].values
y = raw_data.iloc[:, -1].values

for i in range(0, len(y)):
    if y[i] >= 1:
        y[i] = 1

# Dealing with missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X)
X = imputer.transform(X)

print(type(X))
print(X)

New = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13]])
X = np.vstack([X,New])

print(type(X))
print(X)


# min_max_scaler = preprocessing.MinMaxScaler()
# min_max = min_max_scaler.fit_transform(X)
# print(type(min_max))
# print(min_max)
