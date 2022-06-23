import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import recall_score, make_scorer
from sklearn.metrics import plot_confusion_matrix
import joblib
import os
import json

projectDirPath = os.path.abspath('../')


X_train = pd.read_csv(projectDirPath + "/data/cleaned/X_train.csv").values
X_test = pd.read_csv(projectDirPath + "/data/cleaned/X_test.csv").values
y_train = pd.read_csv(projectDirPath + "/data/cleaned/y_train.csv").values.reshape(-1,)
y_test = pd.read_csv(projectDirPath + "/data/cleaned/y_test.csv").values.reshape(-1,)

knn1 = KNeighborsClassifier(n_neighbors = 3)
knn2 = KNeighborsClassifier(n_neighbors = 4)
knn3 = KNeighborsClassifier(n_neighbors = 5)

knn1.fit(X_train, y_train)
knn2.fit(X_train, y_train)
knn3.fit(X_train, y_train)

plot_confusion_matrix(knn1, X_test, y_test)
plot_confusion_matrix(knn2, X_test, y_test)
plot_confusion_matrix(knn3, X_test, y_test)

y_pred1 = knn1.predict(X_test)
y_pred2 = knn2.predict(X_test)
y_pred3 = knn3.predict(X_test)

acc1 = accuracy_score(y_test, y_pred1)
acc2 = accuracy_score(y_test, y_pred2)
acc3 = accuracy_score(y_test, y_pred3)

rec1 = recall_score(y_test, y_pred1)
rec2 = recall_score(y_test, y_pred2)
rec3 = recall_score(y_test, y_pred3)

cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)

print("Accuracy on the test set for 3 neighbors: ", acc1)
print("recall1 on the test set: ", rec1)
print("confusion matrix:\n ", cm1, "\n")

print("Accuracy on the test set for 4 neighbors: ", acc2)
print("recall1 on the test set: ", rec2)
print("confusion matrix:\n ", cm2, "\n")

print("Accuracy on the test set for 5 neighbors: ", acc3)
print("recall1 on the test set: ", rec3)
print("confusion matrix:\n ", cm3)

joblib.dump(knn2, open(projectDirPath + "/models/knn.joblib", 'wb'))

knn1EvaluationData = {"acc": acc1, "rec": rec1, "tn" : int(cm1[0, 0]), "fn" : int(cm1[1, 0]), "tp" : int(cm1[1, 1]), "fp" : int(cm1[0, 1])}
knn2EvaluationData = {"acc": acc2, "rec": rec2, "tn" : int(cm2[0, 0]), "fn" : int(cm2[1, 0]), "tp" : int(cm2[1, 1]), "fp" : int(cm2[0, 1])}
knn3EvaluationData = {"acc": acc3, "rec": rec3, "tn" : int(cm3[0, 0]), "fn" : int(cm3[1, 0]), "tp" : int(cm3[1, 1]), "fp" : int(cm3[0, 1])}

knnEvaluationData = [knn1EvaluationData, knn2EvaluationData, knn3EvaluationData]

i = 1
for data in knnEvaluationData:
    with open(projectDirPath + "\\models\\knn{}.json".format(i), "w") as file:
        json.dump(data, file)
    i += 1


