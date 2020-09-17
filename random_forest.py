import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix

with open("UCI HAR Dataset/features.txt","r") as ft:
    features = [line.strip("\n") for line in ft.readlines()]

#Loading the data:
X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep = '\s+', names = features)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", sep = '\s+', names = ['Activity'])
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep = '\s+', names = features)
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", sep = '\s+', names = ['Activity'])

#Fitting the model:
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy score:",accuracy_score(y_test,y_pred))

labels = ['WALKING', 'WALKING_UP_STAIRS', 'WALKING_DOWN_STAIRS', 'SITTING', 'STANDING', 'LAYING_DOWN']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
cnf_fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels,
                    title='Confusion matrix for Random Forest')

# Plot normalized confusion matrix
normalized_fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                    title='Normalized confusion matrix for Random Forest')
plt.show()
cnf_fig.savefig('./Plots/randomforests_conf_matrix.png')
normalized_fig.savefig('./Plots/randomforest_normalized_conf_matrix')
