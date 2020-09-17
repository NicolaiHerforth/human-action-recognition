import pandas as pd

with open("UCI HAR Dataset/features.txt","r") as ft:
    features = [line.split()[1] for line in ft.readlines()]

X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep = '\s+', names = features)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", sep = '\s+', names = ['Activity'])
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep = '\s+', names = features)
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", sep = '\s+', names = ['Activity'])

#Deleting row 2956 in y and 7531 in X since they contain a Nan in Subject.
# X_train.drop([7351], inplace = True)
# y_train.drop([2946], inplace = True)
# X_test.dropna(how = 'any')
# y_test.drop([2946], inplace = True)

#Making 4 CSVs:
X_train.to_csv("Preprocessed Data/X_train.csv", index=False)
y_train.to_csv("Preprocessed Data/y_train.csv", index=False)
X_test.to_csv("Preprocessed Data/X_test.csv", index=False)
y_test.to_csv("Preprocessed Data/y_test.csv", index=False)