import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle 


X_train = pd.read_csv("data/X_train.csv").values
X_test = pd.read_csv("data/X_test.csv").values
y_train = pd.read_csv("data/y_train.csv").values.reshape(-1,)
y_test = pd.read_csv("data/y_test.csv").values.reshape(-1,)

knn1 = KNeighborsClassifier(n_neighbors = 3)
knn2 = KNeighborsClassifier(n_neighbors = 4)
knn3 = KNeighborsClassifier(n_neighbors = 5)

knn1.fit(X_train, y_train)
knn2.fit(X_train, y_train)
knn3.fit(X_train, y_train)

y_pred1 = knn1.predict(X_test)
y_pred2 = knn2.predict(X_test)
y_pred3 = knn3.predict(X_test)

acc1 = accuracy_score(y_test, y_pred1)
acc2 = accuracy_score(y_test, y_pred2)
acc3 = accuracy_score(y_test, y_pred3)

cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)

print("Accuracy on the test set for 3 neighbors: ", acc1)
print("confusion matrix:\n ", cm1, "\n")

print("Accuracy on the test set for 4 neighbors: ", acc2)
print("confusion matrix:\n ", cm2, "\n")

print("Accuracy on the test set for 5 neighbors: ", acc3)
print("confusion matrix:\n ", cm3)

pickle.dump(knn3, open("models/knn.pkl", 'wb'))