import sklearn
import numpy as np
import sklearn.neighbors as classifier
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

# load our data
X_train_set = np.genfromtxt("UCI HAR Dataset/train/X_train.txt")
y_train_set = np.genfromtxt("UCI HAR Dataset/train/y_train.txt")
X_test_set = np.genfromtxt("UCI HAR Dataset/test/X_test.txt")
y_test_set = np.genfromtxt("UCI HAR Dataset/test/y_test.txt")

# test which parameter is best
first = 0
second = 0
final_results = []
nu_neighbors = 0
for i in range(1,12):
    print("Neighbors = ",i)
    kartoffel = KNN(n_neighbors=i)
    scores = cross_val_score(kartoffel, X_train_set, y_train_set, cv=10, n_jobs=-1)
    result = scores.mean()
    final_results.append(result)
    if result >= first:
        second = first
        nu_neighbors_2 = nu_neighbors
        first = result  
        nu_neighbors = i
        print(" NEW FIRST: ",i)
    elif result >= second:
        second = result
        nu_neighbors_2 = i
        print(" NEW SECOND:",i)
    else:
        continue


print("Optimal was {} with a mean value of {}.".format(nu_neighbors,first))
print("Second was {} with a mean value of {}.".format(nu_neighbors_2,second))
   
# do final test based on optimal amount of neighbors and predict
KNN = KNN(n_neighbors=nu_neighbors)
KNN.fit(X_train_set,y_train_set)
prediction = KNN.predict(X_test_set)

neighbors = ['1','2','3','4','5','6','7','8','9','10','11']
plt.plot(neighbors, final_results)
plt.ylabel("Score")
plt.xlabel("Neighbor count")
plt.show()

# print classification report with correct labels.
labels = ['WALKING', 'WALKING_UP_STAIRS', 'WALKING_DOWN_STAIRS', 'SITTING', 'STANDING', 'LAYING_DOWN']
print(classification_report(y_test_set,prediction, target_names=labels))

# Compute confusion matrix
cm = confusion_matrix(y_test_set, prediction)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
fig = plt.figure()
plot_confusion_matrix(cm, classes=labels, title='KNN confusion matrix')
# Normalize the confusion matrix by row (i.e by the number of samples

# in each class)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print('Normalized confusion matrix')
# print(cm_normalized)
fig_normalized = plt.figure()
plot_confusion_matrix(cm, classes=labels, title='KNN Normalized confusion matrix', normalize=True)

plt.show()
fig.savefig('./Plots/KNN_cnf_matrix.png')
fig_normalized.savefig("./Plots/KNN_normalized_cnf_matrix.png")
 