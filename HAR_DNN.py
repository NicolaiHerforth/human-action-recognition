#import tensorflow as tf
import tflearn
import pandas as pd
import numpy as np
from tflearn.data_utils import load_csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from plot_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
from tflearn.data_utils import to_categorical
'''
merge X and Y train files
# df = pd.read_csv('./Preprocessed Data/X_train.csv', header=0) 
# l = pd.read_csv('./Preprocessed Data/y_train.csv', header=0)
# df.insert(loc=0, column = 'Activity', value=l.values)
# df.to_csv('./Preprocessed Data/train.csv', index = False)
'''

# load train.csv as input file.
# 7 classes due to tensorflow using 0-based indexing. Transforming labels into a 1 x 7 matrix
data, labels = load_csv('./Preprocessed Data/train.csv', target_column=0, categorical_labels=True,n_classes=7)
data = np.array(data, dtype=np.float64)
labels = np.array(labels, dtype=np.int)
class_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
input_layer = tflearn.input_data(shape= [None, 561])
# 1st hidden layer
dense1 = tflearn.fully_connected(input_layer, 64)
# dropout layer to prevent overfitting
dropout1= tflearn.dropout(dense1, 0.5)
# 2nd hidden layer
dense2 = tflearn.fully_connected(dropout1, 64)
dropout2 = tflearn.dropout(dense2, 0.5)
# output layer using softmax
softmax = tflearn.fully_connected(dropout2, 7, activation='softmax')
#net = tflearn.regression(softmax, to_one_hot=True, n_classes=6, loss='categorical_crossentropy')
net = tflearn.regression(softmax, loss='categorical_crossentropy')
# builds a DNN model
model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir='./DNN_log/')
# train our model with train set with 20% as validation set.
model.fit(data, labels, n_epoch = 50, batch_size = 16, show_metric = True, validation_set = 0.2, shuffle=True)

# read X_test and Y_test
X_test = pd.read_csv('./Preprocessed Data/X_test.csv', header=0)
Y_test = pd.read_csv('./Preprocessed Data/y_test.csv', header=0)

# predict the results using our trained model
Y_pred = model.predict_label(X_test)
Y_pred = Y_pred[:,0].tolist()

print(classification_report(Y_test, Y_pred))

cnf_matrix = confusion_matrix(Y_test, Y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
cnf_fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                    title='Confusion matrix for DNN')

# Plot normalized confusion matrix
normalized_fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                    title='Normalized confusion matrix for DNN')
plt.show()
cnf_fig.savefig('./Plots/DNN_conf_matrix.png')
normalized_fig.savefig('./Plots/DNN_normalized_conf_matrix.png')