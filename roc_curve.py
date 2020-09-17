import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

with open("UCI HAR Dataset/features.txt","r") as ft:
    features = [line.strip("\n") for line in ft.readlines()]

X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep = '\s+', names = features)
X_test =  pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep = '\s+', names = features)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", sep = '\s+', names = ['Activity'])
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", sep = '\s+', names = ['Activity'])

#Turn dataframes into numpy arrays:
X_train_t = X_train.as_matrix()
X_test_t = X_test.as_matrix()
y_train_t = y_train.as_matrix()
y_test_t = y_test.as_matrix()

# Binarize the output
y_train_bin = label_binarize(y_train_t, classes=[1,2,3,4,5,6])
y_test_bin = label_binarize(y_test_t, classes=[1,2,3,4,5,6])
n_classes = y_test_bin.shape[1]

#Finding the optimal number of components for the PCA:
pca = PCA(n_components=None)
pca.fit(X_train_t,y_train_bin)

def select_n_components(var_ratio,goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0
    
    # Set initial number of features
    n_components = 0
    
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        
        # Add the explained variance to the total
        total_variance += explained_variance
        
        # Add one to the number of components
        n_components += 1
        
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
            
    # Return the number of components
    return n_components

n_components = select_n_components(pca.explained_variance_ratio_, 0.9999)

#Run the PCA with the optimal number of components:
pca2 = PCA(n_components = n_components)
pca2.fit(X_train_t ,y_test_bin)

#Transforming the matrices:
X_train2 = pca2.transform(X_train_t)
X_test2 = pca2.transform(X_test_t)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True,
                                 random_state=0))
y_score = classifier.fit(X_train2, y_train_bin).decision_function(X_test2)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['darkorange', 'cornflowerblue', 'crimson','royalblue','coral','lightskyblue'])
labels = ['Walking','WalkingUpstairs','WalkingDownstairs','Sitting','Standing','Laying']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(labels[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("Plots/roc_curve_svm.png")
plt.show()