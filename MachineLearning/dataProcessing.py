import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
import os

projectDirPath = os.path.abspath('../')

#replace every ? with np.nan in dataset  
raw_data = pd.read_csv(projectDirPath + "/data/raw/mixed_data.data", sep = ',', header=None)
raw_data = raw_data.replace('?', np.nan)

#splitting classes indicators from rest of the data
X = raw_data.iloc[:, :-1].values
y = raw_data.iloc[:, -1].values

for i in range(0, len(y)):
    if y[i] >= 1:
        y[i] = 1
        

#Dealing with missing data
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(X)
X = imputer.transform(X)

X_transformed = pd.DataFrame(X)
X_transformed.to_csv(projectDirPath + "/data/cleaned/X_cleaned.csv", index = False)

min_max_scaler = preprocessing.MinMaxScaler()
min_max = min_max_scaler.fit_transform(X)

#Spliting dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(min_max, y, test_size = 0.2)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)


X_train.to_csv(projectDirPath + "/data/cleaned/X_train.csv", index = False)
X_test.to_csv(projectDirPath + "/data/cleaned/X_test.csv", index = False)
y_train.to_csv(projectDirPath + "/data/cleaned/y_train.csv", index = False)
y_test.to_csv(projectDirPath + "/data/cleaned/y_test.csv", index = False)