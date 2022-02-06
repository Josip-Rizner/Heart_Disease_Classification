import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

#dealing with ? in data
raw_data = pd.read_csv("MachineLearning/data/cleveland.data", sep = ',', header=None)
raw_data = raw_data.replace('?', np.nan)


#Sepparating classes from other data
X = raw_data.iloc[:, :-1].values
y = raw_data.iloc[:, -1].values


#Dealing with missing data
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(X)
X = imputer.transform(X)


#Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


#Saving training and test sets
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

X_train.to_csv("MachineLearning/data/X_train.csv", index = False)
X_test.to_csv("MachineLearning/data/X_test.csv", index = False)
y_train.to_csv("MachineLearning/data/y_train.csv", index = False)
y_test.to_csv("MachineLearning/data/y_test.csv", index = False)





