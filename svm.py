import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from plot_confusion_matrix import plot_confusion_matrix

with open("UCI HAR Dataset/features.txt","r") as ft:
    features = [line.strip("\n") for line in ft.readlines()]

X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep = '\s+', names = features)
X_test =  pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep = '\s+', names = features)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", sep = '\s+', names = ['Activity'])
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", sep = '\s+', names = ['Activity'])

clf = svm.LinearSVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

labels = ['WALKING', 'WALKING_UP_STAIRS', 'WALKING_DOWN_STAIRS', 'SITTING', 'STANDING', 'LAYING_DOWN']
print(classification_report(y_test,pred, target_names=labels))

#Crossvalidation
SVC_tol_def = svm.SVC(max_iter=5000)
SVC_tol_1 = svm.SVC(tol = 1e-2, max_iter=5000)
SVC_tol_2 = svm.SVC(tol = 1e-4,max_iter=5000)

lin_tol_def = svm.LinearSVC(max_iter=5000)
lin_tol_1 = svm.LinearSVC(tol=1e-3,max_iter=5000)
lin_tol_2 = svm.LinearSVC(tol=1e-5,max_iter=5000)

methods = [SVC_tol_def,SVC_tol_1,SVC_tol_2,lin_tol_def, lin_tol_1, lin_tol_2]
kernels = ["SVC_tol_def","SVC_tol_1","SVC_tol_2","lin_tol_def", "lin_tol_1", "lin_tol_2"]
results = []
for i in methods:
    print("Cross validation on ", i)
    scores = cross_val_score(i,X_train, y_train, cv = 10,n_jobs = -1)
    results.append(scores.mean())
    print(results)

plt.ylabel("Accuracy Score")
plt.xlabel("Parameters")
plt.plot(kernels,results)
plt.show()

"""
# Compute confusion matrix
cm = confusion_matrix(y_test, pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
fig = plt.figure()
plot_confusion_matrix(cm, classes=labels, title='SVM confusion matrix')

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print('Normalized confusion matrix')
# print(cm_normalized)
fig_normalized = plt.figure()
plot_confusion_matrix(cm, classes=labels, title='SVM Normalized confusion matrix', normalize=True)

plt.show

fig.savefig('./Plots/SVM_cnf_matrix.png')
fig_normalized.savefig('./Plots/SVM_normalized_cnf_matrix.png')
"""