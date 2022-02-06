import pandas as pd
import numpy as np

X_train = pd.read_csv("MachineLearningX_train.csv").values
X_test = pd.read_csv("ready data\\X_test.csv").values
y_train = pd.read_csv("ready data\\y_train.csv").values.reshape(-1,)
y_test = pd.read_csv("\\ready data\\y_test.csv").values.reshape(-1,)