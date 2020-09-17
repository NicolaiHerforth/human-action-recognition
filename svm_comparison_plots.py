import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt

with open("UCI HAR Dataset/features.txt","r") as ft:
    features = [line.strip("\n") for line in ft.readlines()]

#Loading the data:
X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep = '\s+', names = features)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", sep = '\s+', names = ['Activity'])
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep = '\s+', names = features)
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", sep = '\s+', names = ['Activity'])
   
#model = RandomForestClassifier(n_estimators=200)
#model = KNN(n_neighbors=3)
model = SVC(kernel="linear",C=1)
#model = LogisticRegression()
#model = ExtraTreesClassifier()

#Computing the recall, precision and F1 score using all features:
model.fit(X_train,y_train)
pred = model.predict(X_test)
precision, recall, fscore, support = score(y_test, pred)
precision_rest = 1-precision
recall_rest = 1-recall
fscore_rest = 1-fscore

#Finding the optimal number of components for the PCA:
pca = PCA(n_components=None)
pca.fit(X_train,y_train)

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
pca2 = PCA(n_components=n_components)
pca2.fit(X_train,y_train)

#Transforming the matrices:
X_train2 = pca2.transform(X_train)
X_test2 = pca2.transform(X_test)

#Computing the recall, precision and F1 score using the selected features:
model.fit(X_train2,y_train)
pred2 = model.predict(X_test2)
precision2, recall2, fscore2, support2 = score(y_test, pred2)
precision2_rest = 1-precision2
recall2_rest = 1-recall2
fscore2_rest = 1-fscore2

#Plot:
N = 6
ind = np.arange(N) # the x locations for the classes
width = 0.3 # the width of the bars: can also be len(x) sequence

f, axarr = plt.subplots(1, 2)
axarr[0].bar(ind, precision, width,label='Precision')
axarr[0].bar(ind, precision_rest, width, bottom=precision2,label='All data')
axarr[0].bar(ind+width, recall, width, color='lightskyblue',label='Recall')
axarr[0].bar(ind+width, recall_rest, width, bottom=recall2, color='crimson',
     label='All data')
axarr[0].bar(ind+width+width, fscore, width, color='mediumblue',label='F1')
axarr[0].bar(ind+width+width, fscore_rest, width, bottom=fscore, color='lightcoral',
     label='All data')
axarr[0].set_title('Using all features')
axarr[0].set(xlabel='Class')
axarr[0].set(ylabel='Scores')
axarr[1].bar(ind, precision2, width, label='Precision')
axarr[1].bar(ind, precision2_rest, width, bottom=precision2, label='All data')
axarr[1].bar(ind+width, recall2, width, color='lightskyblue', label='Recall')
axarr[1].bar(ind+width, recall2_rest, width, bottom=recall2, color='crimson', label='All data')
axarr[1].bar(ind+width+width, fscore2, width, color='mediumblue',label='F1')
axarr[1].bar(ind+width+width, fscore2_rest, width, bottom=fscore2, color='lightcoral',
     label='All data')
axarr[1].set_title('Using selected features')
axarr[1].set(xlabel='Class')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axarr.flat:
    ax.label_outer()
    ax.set_xticks(ind + (2*width) / 3)
    ax.set_xticklabels(['Walking','WalkingUpstairs','WalkingDownstairs','Sitting','Standing','Laying'], rotation=40, ha='right')
    ax.set_yticks(np.arange(0, 1.3, 0.2))

f.tight_layout()

#plt.xticks(rotation = 40)
plt.legend()
plt.savefig("Plots/svm_comparison.png")