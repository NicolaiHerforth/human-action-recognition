import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from plot_ROC_curve import plot_ROC_curve
from sklearn.model_selection import GridSearchCV


with open("UCI HAR Dataset/features.txt","r") as ft:
    features = [line.split()[1] for line in ft.readlines()]

X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep = '\s+', names = features)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", sep = '\s+', names = ['activity'])
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep = '\s+', names = features)
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", sep = '\s+', names = ['label'])

# convert dataframe to numpy matrix
X_train_t = X_train.as_matrix()
y_train_t = y_train.as_matrix()
X_test_t = X_test.as_matrix()
y_test_t = y_test.as_matrix()

# binarize labels
y_train_bin = label_binarize(y_train_t, classes=[1,2,3,4,5,6])
y_test_bin = label_binarize(y_test_t, classes=[1,2,3,4,5,6])
n_classes = y_test_bin.shape[1]

class_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
#kfold = model_selection.KFold(n_splits=10)
kfold = model_selection.StratifiedKFold(n_splits=10)
estm1 = []
clf1 = LogisticRegression(random_state=20)
estm1.append(('logistic', clf1))
clf2 = SVC(kernel='linear', probability=True, random_state=20)
estm1.append(('svc', clf2))
clf3 = KNN(n_neighbors=10)
estm1.append(('knn', clf3))

eclf = VotingClassifier(estm1, n_jobs=-1, voting='soft', weights=[1.6, 3, 1])
# multiclass classifer
classifier = OneVsRestClassifier(eclf)
# train ensemble
eclf.fit(X_train, y_train['activity'].ravel())
# predict probabilities of each class
y_score = classifier.fit(X_train_t, y_train_bin).predict_proba(X_test_t)
y_pred = eclf.predict(X_test).tolist()
y_true = y_test['label'].values.tolist()
#score_cv = model_selection.cross_val_score(eclf, X_train, y_train['activity'].ravel(), cv=kfold)
# y_pred_cv = model_selection.cross_val_predict(eclf, X_train, y_train['activity'].ravel(), cv=kfold)
print(accuracy_score(y_true, y_pred))
#print(score_cv)
print(classification_report(y_true, y_pred))

# TO DO: more combiclf
estimators2 = []

roc_fig = plt.figure()
plot_ROC_curve(y_test_bin, y_score, 6, 'ROC curve for ensemble of logistic, KNN, and SVM')
roc_fig.savefig('./Plots/voting_ensemble_ROC_curve.png')

# # Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
cnf_fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix for voting classifier')

# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
plt.show()
cnf_fig.savefig('./Plots/ensemble_cnf_matrix.png')