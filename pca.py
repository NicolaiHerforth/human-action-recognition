import pandas as pd
from sklearn.decomposition import PCA
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

#Computing the accuracy using all features:
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("Accuracy using all features:",accuracy_score(y_test,pred))

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

#Computing the accuracy using the selected features:
model.fit(X_train2,y_train)
pred2 = model.predict(X_test2)
print("Accuracy using",n_components,"features:",accuracy_score(y_test,pred2))  