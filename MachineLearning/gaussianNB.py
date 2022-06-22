import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import joblib
import os
import json

projectDirPath = os.path.abspath('../')


X_train = pd.read_csv(projectDirPath + "/data/cleaned/X_train.csv").values
X_test = pd.read_csv(projectDirPath + "/data/cleaned/X_test.csv").values
y_train = pd.read_csv(projectDirPath + "/data/cleaned/y_train.csv").values.reshape(-1,)
y_test = pd.read_csv(projectDirPath + "/data/cleaned/y_test.csv").values.reshape(-1,)

gaussianNB = GaussianNB()
gaussianNB.fit(X_train, y_train)


plot_confusion_matrix(gaussianNB, X_test, y_test)
y_pred = gaussianNB.predict(X_test)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("accuracy on the test set: ", acc)
print("recall on the test set: ", rec)
print("confusion matrix:\n ", cm)


joblib.dump(gaussianNB, open(projectDirPath + "/models/gaussianNB.joblib", 'wb'))

gaussianNBEvaluationData = {"acc": acc, "rec": rec, "tn" : int(cm[0, 0]), "fn" : int(cm[1, 0]), "tp" : int(cm[1, 1]), "fp" : int(cm[0, 1])}

with open(projectDirPath + "\\models\\gaussianNB.json", "w") as file:
    json.dump(gaussianNBEvaluationData, file)
