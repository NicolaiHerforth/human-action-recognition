import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

with open("UCI HAR Dataset/features.txt","r") as ft:
    features = [line.strip("\n") for line in ft.readlines()]

#Loading the data:
X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep = '\s+', names = features)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", sep = '\s+', names = ['Activity'])
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep = '\s+', names = features)
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", sep = '\s+', names = ['Activity'])

labels = ['WALKING', 'WALKING_UP_STAIRS', 'WALKING_DOWN_STAIRS', 'SITTING', 'STANDING', 'LAYING_DOWN']
#Using 154 features:
features2 = ['4 tBodyAcc-std()-X', '5 tBodyAcc-std()-Y', '7 tBodyAcc-mad()-X', '10 tBodyAcc-max()-X', '13 tBodyAcc-min()-X', '23 tBodyAcc-entropy()-X', '24 tBodyAcc-entropy()-Y', '25 tBodyAcc-entropy()-Z', '29 tBodyAcc-arCoeff()-X,4', '33 tBodyAcc-arCoeff()-Y,4', '37 tBodyAcc-arCoeff()-Z,4', '38 tBodyAcc-correlation()-X,Y', '40 tBodyAcc-correlation()-Y,Z', '41 tGravityAcc-mean()-X', '42 tGravityAcc-mean()-Y', '43 tGravityAcc-mean()-Z', '44 tGravityAcc-std()-X', '45 tGravityAcc-std()-Y', '46 tGravityAcc-std()-Z', '47 tGravityAcc-mad()-X', '48 tGravityAcc-mad()-Y', '50 tGravityAcc-max()-X', '51 tGravityAcc-max()-Y', '52 tGravityAcc-max()-Z', '53 tGravityAcc-min()-X', '54 tGravityAcc-min()-Y', '56 tGravityAcc-sma()', '57 tGravityAcc-energy()-X', '58 tGravityAcc-energy()-Y', '59 tGravityAcc-energy()-Z', '61 tGravityAcc-iqr()-Y', '62 tGravityAcc-iqr()-Z', '63 tGravityAcc-entropy()-X', '64 tGravityAcc-entropy()-Y', '66 tGravityAcc-arCoeff()-X,1', '70 tGravityAcc-arCoeff()-Y,1', '71 tGravityAcc-arCoeff()-Y,2', '73 tGravityAcc-arCoeff()-Y,4', '90 tBodyAccJerk-max()-X', '92 tBodyAccJerk-max()-Z', '93 tBodyAccJerk-min()-X', '103 tBodyAccJerk-entropy()-X', '104 tBodyAccJerk-entropy()-Y', '105 tBodyAccJerk-entropy()-Z', '106 tBodyAccJerk-arCoeff()-X,1', '110 tBodyAccJerk-arCoeff()-Y,1', '113 tBodyAccJerk-arCoeff()-Y,4', '115 tBodyAccJerk-arCoeff()-Z,2', '118 tBodyAccJerk-correlation()-X,Y', '119 tBodyAccJerk-correlation()-X,Z', '133 tBodyGyro-min()-X', '142 tBodyGyro-iqr()-Z', '143 tBodyGyro-entropy()-X', '146 tBodyGyro-arCoeff()-X,1', '148 tBodyGyro-arCoeff()-X,3', '149 tBodyGyro-arCoeff()-X,4', '150 tBodyGyro-arCoeff()-Y,1', '153 tBodyGyro-arCoeff()-Y,4', '157 tBodyGyro-arCoeff()-Z,4', '158 tBodyGyro-correlation()-X,Y', '159 tBodyGyro-correlation()-X,Z', '160 tBodyGyro-correlation()-Y,Z', '167 tBodyGyroJerk-mad()-X', '170 tBodyGyroJerk-max()-X', '180 tBodyGyroJerk-iqr()-X', '182 tBodyGyroJerk-iqr()-Z', '183 tBodyGyroJerk-entropy()-X', '184 tBodyGyroJerk-entropy()-Y', '185 tBodyGyroJerk-entropy()-Z', '186 tBodyGyroJerk-arCoeff()-X,1', '187 tBodyGyroJerk-arCoeff()-X,2', '188 tBodyGyroJerk-arCoeff()-X,3', '190 tBodyGyroJerk-arCoeff()-Y,1', '191 tBodyGyroJerk-arCoeff()-Y,2', '193 tBodyGyroJerk-arCoeff()-Y,4', '194 tBodyGyroJerk-arCoeff()-Z,1', '195 tBodyGyroJerk-arCoeff()-Z,2', '198 tBodyGyroJerk-correlation()-X,Y', '199 tBodyGyroJerk-correlation()-X,Z', '200 tBodyGyroJerk-correlation()-Y,Z', '202 tBodyAccMag-std()', '203 tBodyAccMag-mad()', '215 tGravityAccMag-std()', '223 tGravityAccMag-arCoeff()1', '234 tBodyAccJerkMag-iqr()', '235 tBodyAccJerkMag-entropy()', '244 tBodyGyroMag-min()', '247 tBodyGyroMag-iqr()', '248 tBodyGyroMag-entropy()', '269 fBodyAcc-std()-X', '270 fBodyAcc-std()-Y', '272 fBodyAcc-mad()-X', '275 fBodyAcc-max()-X', '276 fBodyAcc-max()-Y', '277 fBodyAcc-max()-Z', '287 fBodyAcc-iqr()-Z', '288 fBodyAcc-entropy()-X', '296 fBodyAcc-meanFreq()-Z', '298 fBodyAcc-kurtosis()-X', '299 fBodyAcc-skewness()-Y', '303 fBodyAcc-bandsEnergy()-1,8', '338 fBodyAcc-bandsEnergy()-57,64', '349 fBodyAccJerk-std()-Y', '355 fBodyAccJerk-max()-Y', '366 fBodyAccJerk-iqr()-Z', '367 fBodyAccJerk-entropy()-X', '368 fBodyAccJerk-entropy()-Y', '369 fBodyAccJerk-entropy()-Z', '370 fBodyAccJerk-maxInds-X', '372 fBodyAccJerk-maxInds-Z', '375 fBodyAccJerk-meanFreq()-Z', '382 fBodyAccJerk-bandsEnergy()-1,8', '383 fBodyAccJerk-bandsEnergy()-9,16', '387 fBodyAccJerk-bandsEnergy()-41,48', '389 fBodyAccJerk-bandsEnergy()-57,64', '417 fBodyAccJerk-bandsEnergy()-57,64', '434 fBodyGyro-max()-Y', '435 fBodyGyro-max()-Z', '446 fBodyGyro-entropy()-X', '447 fBodyGyro-entropy()-Y', '448 fBodyGyro-entropy()-Z', '450 fBodyGyro-maxInds-Y', '451 fBodyGyro-maxInds-Z', '452 fBodyGyro-meanFreq()-X', '458 fBodyGyro-kurtosis()-Y', '459 fBodyGyro-skewness()-Z', '460 fBodyGyro-kurtosis()-Z', '464 fBodyGyro-bandsEnergy()-25,32', '466 fBodyGyro-bandsEnergy()-41,48', '474 fBodyGyro-bandsEnergy()-25,48', '478 fBodyGyro-bandsEnergy()-25,32', '479 fBodyGyro-bandsEnergy()-33,40', '482 fBodyGyro-bandsEnergy()-57,64', '490 fBodyGyro-bandsEnergy()-9,16', '491 fBodyGyro-bandsEnergy()-17,24', '492 fBodyGyro-bandsEnergy()-25,32', '498 fBodyGyro-bandsEnergy()-17,32', '502 fBodyGyro-bandsEnergy()-25,48', '503 fBodyAccMag-mean()', '505 fBodyAccMag-mad()', '508 fBodyAccMag-sma()', '509 fBodyAccMag-energy()', '510 fBodyAccMag-iqr()', '517 fBodyBodyAccJerkMag-std()', '524 fBodyBodyAccJerkMag-entropy()', '527 fBodyBodyAccJerkMag-skewness()', '533 fBodyBodyGyroMag-min()', '538 fBodyBodyGyroMag-maxInds', '539 fBodyBodyGyroMag-meanFreq()', '550 fBodyBodyGyroJerkMag-entropy()', '551 fBodyBodyGyroJerkMag-maxInds', '554 fBodyBodyGyroJerkMag-kurtosis()', '559 angle(X,gravityMean)', '560 angle(Y,gravityMean)']
X_train2 = X_train[features2]
X_test2 = X_test[features2]

model = LogisticRegression()
model.fit(X_train2,y_train)
#model.fit(X_train,y_train)

#pred = model.predict(X_test)
y_pred = model.predict(X_test2)
print(classification_report(y_test,y_pred, target_names=labels))
print(accuracy_score(y_test, y_pred))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
cnf_fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels,
                    title='Confusion matrix for logisitc regression')

# Plot normalized confusion matrix
normalized_fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                    title='Normalized confusion matrix for logisitic regression')
plt.show()
cnf_fig.savefig('./Plots/logisitic_conf_matrix.png')
normalized_fig.savefig('./Plots/logistic_normalized_conf_matrix.png')

