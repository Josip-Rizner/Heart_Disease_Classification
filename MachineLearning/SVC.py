import pandas as pd
from sklearn.svm import SVC
#from sklearn.model_selection import cross_val_score
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

#svc = SVC(kernel = "rbf", C=2, degree=2)
svc = SVC(kernel = "poly", degree=2)
svc.fit(X_train, y_train)

"""
accuracies = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 10)
recall_scorer = make_scorer(recall_score)
recalls = cross_val_score(estimator = svc, X = X_train, y = y_train, scoring = recall_scorer, cv = 10)
accMean = accuracies.mean()
accStdDev = accuracies.std()
recMean = recalls.mean()
recStdDev = recalls.std()
print("mean of 10 accuracies: ", accMean)
print("standard deviation of accuracies: ", recStdDev)
print("mean of 10 recalls: ", accMean)
print("standard deviation of recalls: ", recStdDev)
"""


plot_confusion_matrix(svc, X_test, y_test)
y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("accuracy on the test set: ", acc)
print("recall on the test set: ", rec)
print("confusion matrix:\n ", cm)


joblib.dump(svc, open(projectDirPath + "/models/svc.joblib", 'wb'))


svcEvaluationData = {"acc": acc, "rec": rec, "tn" : int(cm[0, 0]), "fn" : int(cm[1, 0]), "tp" : int(cm[1, 1]), "fp" : int(cm[0, 1])}

with open(projectDirPath + "\\models\\svc.json", "w") as file:
    json.dump(svcEvaluationData, file)