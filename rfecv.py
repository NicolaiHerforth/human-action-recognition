import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier as KNN
#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

with open("UCI HAR Dataset/features.txt","r") as ft:
    features = [line.strip("\n") for line in ft.readlines()]

#Loading the data:
X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep = '\s+', names = features)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", sep = '\s+', names = ['Activity'])
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep = '\s+', names = features)
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", sep = '\s+', names = ['Activity'])

# Choosing the model:
#model = RandomForestClassifier(n_estimators=200)
#model = KNN(n_neighbors=3)
#model = SVC(kernel="linear",C=1)
#model = LogisticRegression()
model = ExtraTreesClassifier()

#Using all features:
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("Accuracy for all features:",accuracy_score(y_test,pred))

# Using RFECV and Stratified Kfold to pick best features:
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, y_train)

#Print the optimal number of features:
print("Optimal number of features : %d" % rfecv.n_features_)

#Plot number of features vs. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig(("Plots/n_features_et.png"))

#Making new versions of X containing only the chosen features:
X_train2 = rfecv.transform(X_train)
X_test2 = rfecv.transform(X_test)

#Using only chosen features:
model.fit(X_train2,y_train)
pred = model.predict(X_test2)
print("Accuracy for chosen features:",accuracy_score(y_test,pred))