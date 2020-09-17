import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
This script makes boxplots of 5 different features and their correlation with 
the activity names.
'''

with open("UCI HAR Dataset/features.txt","r") as ft:
    features = [line.split()[1] for line in ft.readlines()]

#Loading the data:
X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep = '\s+', names = features)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", sep = '\s+', names = ['Activity'])
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep = '\s+', names = features)
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", sep = '\s+', names = ['Activity'])

#Adding subject and activity:
X_train['Subject'] = pd.read_csv("UCI HAR Dataset/train/subject_train.txt")
X_test['Subject'] = pd.read_csv("UCI HAR Dataset/train/subject_train.txt")
X_train['Activity'] = y_train['Activity']

#Deleting row 2956 in y and 7531 in X since they contain a Nan in Subject.
X_train.drop([7351], inplace = True)
y_train.drop([2946], inplace = True)
X_test.dropna(how = 'any')
y_test.drop([2946], inplace = True)

#Adding activity names:
X_train['ActivityNames'] = X_train['Activity']
X_train['ActivityNames'] = X_train['ActivityNames'].replace(1,'Walking')
X_train['ActivityNames'] = X_train['ActivityNames'].replace(2,'WalkingUpstairs')
X_train['ActivityNames'] = X_train['ActivityNames'].replace(3,'WalkingDownstairs')
X_train['ActivityNames'] = X_train['ActivityNames'].replace(4,'Sitting')
X_train['ActivityNames'] = X_train['ActivityNames'].replace(5,'Standing')
X_train['ActivityNames'] = X_train['ActivityNames'].replace(6,'Laying')

#Adding subject, activity and activity names to the feature list as well:
features = features + ['Subject', 'Activity', 'ActivityNames']

print("Started plotting")

plt.figure()
#sns.pairplot(data = X_train[['angle(X,gravityMean)','tGravityAcc-mean()-Y',
#                             'tGravityAcc-energy()-X','tGravityAcc-max()-Y',
#                             'tGravityAcc-max()-X','ActivityNames']], hue = 'ActivityNames')
#sns.pairplot(data = X_train[['fBodyAcc-std()-X','tBodyAccJerk-entropy()-X',
#                             'tGravityAcc-mean()-Y','tBodyAccJerkMag-iqr()',
#                             'tGravityAccMag-std()','ActivityNames']], hue = 'ActivityNames')
sns.pairplot(data = X_train[['tGravityAccMag-std()','tGravityAcc-energy()-X',
                             'tGravityAcc-max()-Y','tBodyAcc-correlation()-X,Y',
                             'tBodyGyroJerk-entropy()-X','ActivityNames']], hue = 'ActivityNames')
plt.savefig("Plots/pairplot3.png")