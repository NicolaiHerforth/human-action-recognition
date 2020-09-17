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
features = features + ['Subject', 'Activity', 'ActivityName']

#Making the figure:
fig, axs = plt.subplots(ncols=5)
fig.tight_layout()

#Making the 5 boxplots:
one = sns.boxplot(x='ActivityNames', y='angle(X,gravityMean)', data = X_train, showfliers = False, saturation = 1, ax=axs[0])
two = sns.boxplot(x='ActivityNames', y='tGravityAcc-mean()-Y', data = X_train, showfliers = False, saturation = 1, ax=axs[1])
three = sns.boxplot(x='ActivityNames', y='tGravityAcc-energy()-X', data = X_train, showfliers = False, saturation = 1, ax=axs[2])
four = sns.boxplot(x='ActivityNames', y='tGravityAcc-max()-Y', data = X_train, showfliers = False, saturation = 1, ax=axs[3])
five = sns.boxplot(x='ActivityNames', y='tGravityAcc-max()-X', data = X_train, showfliers = False, saturation = 1, ax=axs[4])

#Making the range of the y-axis the same in all boxplots:
one.set(ylim=(-1.1, 1.1))
two.set(ylim=(-1.1, 1.1))
three.set(ylim=(-1.1, 1.1))
four.set(ylim=(-1.1, 1.1))
five.set(ylim=(-1.1, 1.1))

#Deleting the y-axis tick labels except on the first boxplot: 
two.axes.yaxis.set_ticklabels([])
three.axes.yaxis.set_ticklabels([])
four.axes.yaxis.set_ticklabels([])
five.axes.yaxis.set_ticklabels([])
one.axes.xaxis.set_ticklabels([])
two.axes.xaxis.set_ticklabels([])
three.axes.xaxis.set_ticklabels([])
four.axes.xaxis.set_ticklabels([])
five.axes.xaxis.set_ticklabels([])

#Deleting the y-axis label on all boxplots:
one.set_ylabel('')
two.set_ylabel('')
three.set_ylabel('')
four.set_ylabel('')
five.set_ylabel('')
one.set_xlabel('')
two.set_xlabel('')
three.set_xlabel('')
four.set_xlabel('')
five.set_xlabel('')
    
#Rotating the x-axis tick labels:
for tick in axs[0].get_xticklabels():
    tick.set_rotation(40)
for tick in axs[1].get_xticklabels():
    tick.set_rotation(40)    
for tick in axs[2].get_xticklabels():
    tick.set_rotation(40)    
for tick in axs[3].get_xticklabels():
    tick.set_rotation(40)    
for tick in axs[4].get_xticklabels():
    tick.set_rotation(40)

#Changing the font size:    
one.tick_params(axis='y', labelsize=8)
one.set_title('angle(X,gravityMean)', fontsize=8)
two.set_title('tGravityAcc-mean()-Y', fontsize=8)
three.set_title('tGravityAcc-energy()-X', fontsize=8)
four.set_title('tGravityAcc-max()-Y', fontsize=8)
five.set_title('tGravityAcc-max()-X', fontsize=8)
    
#Saving the figure in the plot directory as "boxplots.png":
plt.savefig("Plots/boxplots.png")