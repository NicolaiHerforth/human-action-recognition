import sklearn
import sklearn.neighbors as classifier
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score

# load our data
X_train_set = np.genfromtxt("UCI HAR Dataset/train/X_train.txt")
y_train_set = np.genfromtxt("UCI HAR Dataset/train/y_train.txt")
X_test_set = np.genfromtxt("UCI HAR Dataset/test/X_test.txt")
y_test_set = np.genfromtxt("UCI HAR Dataset/test/y_test.txt")

# initialise our variables for testing.
print('Initialize variables')
neighbors = [1,2,3,4,5,6,7,8,9,10,11]
runs = 10
score = []

for i in range(1,12):
    print('attempt', i)
    estimator = KNN(n_neighbors=i) 
    scores = cross_val_score(estimator, X_train_set, y_train_set, cv=runs, n_jobs=-1)
    three_avg = sum(scores)/len(scores) 
    three_avg = round(three_avg, 3)
    score.append(three_avg)
    print(score)
    print('Average score in cross validation with', i, ' neighbors:', three_avg, '\n')
print(score)
    
plt.ylabel("Accuracy Score")
plt.xlabel("Neighbor count")
plt.plot(neighbors,score)
plt.show()