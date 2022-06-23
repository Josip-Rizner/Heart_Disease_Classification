import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.preprocessing import PolynomialFeatures
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

logisticRegression = LogisticRegression(C=1000)
logisticRegression.fit(X_train, y_train)



#applying transformation on features to achive non-linear decision border
#transformation = PolynomialFeatures(degree = 2)
#X_train_t = transformation.fit_transform(X_train)
#X_test_t = transformation.fit_transform(X_test)

#training new model on transformed features
#polyLogisticRegression = LogisticRegression(max_iter = 1800)
#polyLogisticRegression.fit(X_train_t, y_train)




#testing of model with linear border on the test set and computing : accuracy, recall and confusion matrix
y_pred = logisticRegression.predict(X_test)
plot_confusion_matrix(logisticRegression, X_test, y_test)

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("accuracy on the test set: ", acc)
print("recall on the test set: ", rec)
print("confusion matrix:\n ", cm)




#testing of model with non-linear border on the test set and computing : accuracy, recall and confusion matrix
"""
y_pred_p = polyLogisticRegression.predict(X_test_t)
plot_confusion_matrix(polyLogisticRegression, X_test_t, y_test)

accP = accuracy_score(y_test, y_pred_p)
recP = recall_score(y_test, y_pred_p)
cmP = confusion_matrix(y_test, y_pred_p)


print("accuracy on the test set: ", accP)
print("recall on the test set: ", recP)
print("confusion matrix:\n ", cmP)
"""



joblib.dump(logisticRegression, open(projectDirPath + "/models/logisticReg.joblib", 'wb'))



logisticRegressionEvaluationData = {"acc": acc, "rec": rec, "tn" : int(cm[0, 0]), "fn" : int(cm[1, 0]), "tp" : int(cm[1, 1]), "fp" : int(cm[0, 1])}
#polyLogisticRegressionEvaluationData = {"acc": accP, "rec": recP, "tn" : int(cmP[0, 0]), "fn" : int(cmP[1, 0]), "tp" : int(cmP[1, 1]), "fp" : int(cmP[0, 1])}

with open(projectDirPath + "\\models\\logisticRegression.json", "w") as file:
    json.dump(logisticRegressionEvaluationData, file)
    
#with open(projectDirPath + "\\models\\polyLogisticRegression.json", "w") as file:
#    json.dump(polyLogisticRegressionEvaluationData, file)


