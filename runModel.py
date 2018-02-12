# coding: utf-8
import numpy as np
import pandas
import functools

import sys

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# clear screen
# import os
# os.system('cls') # on windows

# =================================================================================
# Globals
# =================================================================================
numFeatures = 30
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
np.set_printoptions(precision=2)

# =================================================================================
# Function Definitions
# =================================================================================
import keras.backend as kbe
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def baseline_model():
     # create model
     model = Sequential()
     # Add an input layer
     model.add(Dense(16, input_dim=numFeatures, activation='relu'))

     # Add one hidden layer
     model.add(Dense(8, activation='relu'))

     ## Add an output layer
     model.add(Dense(3, activation='softmax'))

     # Compile model
     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

     return model

import itertools
from itertools import product
import matplotlib.pyplot as plt
def plot_confusion_matrix(imageFile, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(imageFile)
    plt.show()

def plot_roc(y, prob_y, roc_file):
    fig = plt.figure()
    fpr0, tpr0, thr0 = metrics.roc_curve(y, prob_y[:,0], pos_label=0)
    plt.plot(fpr0, tpr0, label='UE')
    print('UE Classifier: TPR at 1%% FPR is %5.3f%%.' % (tpr0[np.argmax(fpr0 > 0.01)] * 100))
    print('UE Classifier: TPR at 5%% FPR is %5.3f%%.' % (tpr0[np.argmax(fpr0 > 0.05)] * 100))
    fpr1, tpr1, thr1 = metrics.roc_curve(y, prob_y[:,1], pos_label=1)
    plt.plot(fpr1, tpr1, label='FE')
    print('FE Classifier: TPR at 1%% FPR is %5.3f%%.' % (tpr1[np.argmax(fpr1 > 0.01)] * 100))
    print('FE Classifier: TPR at 5%% FPR is %5.3f%%.' % (tpr1[np.argmax(fpr1 > 0.05)] * 100))
    fpr2, tpr2, thr2 = metrics.roc_curve(y, prob_y[:,2], pos_label=2)
    plt.plot(fpr2, tpr2, label='W')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    print('W Classifier: TPR at 1%% FPR is %5.3f%%.' % (tpr2[np.argmax(fpr2 > 0.01)] * 100))
    print('W Classifier: TPR at 5%% FPR is %5.3f%%.' % (tpr2[np.argmax(fpr2 > 0.05)] * 100))
    plt.title('ROC')
    plt.legend()
    plt.savefig(roc_file)

def runClassifier(trainFeaturesFile, testFeaturesFile, menOrWomen, runType):
    # Load datasets.
    train_dataset = pandas.read_csv(trainFeaturesFile, float_precision='round_trip')

    # Convert outcome vector to one-hot-encoded matrix
    le = LabelEncoder()
    y_train = train_dataset[['outcome']].values
    le.fit(np.ravel(y_train))
    encoded_y_train = le.transform(np.ravel(y_train))
    y_train_encoded = np_utils.to_categorical(np.ravel(encoded_y_train))
    y_train = encoded_y_train

    # Drop the last column containing outcomes
    del train_dataset['outcome']
    X_train = train_dataset.values

    # =================================================================================
    # Classifier
    # =================================================================================
    numFeatures = X_train.shape[1]
    classifier = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=1000, verbose=1)
    classifier.fit(X_train, y_train_encoded)

    # from sklearn import svm
    # classifier = svm.SVC(C=1.0, gamma=0.1, cache_size=40, class_weight=None, coef0=0.0,
    #                      decision_function_shape='ovr', kernel='rbf',
    #                      max_iter=-1, probability=True, random_state=None, shrinking=True,
    #                      tol=0.001, verbose=True)
    # classifier.fit(X_train, y_train)

    if runType == 'predict':
        test_dataset = pandas.read_csv(testFeaturesFile, float_precision='round_trip')
        del test_dataset['outcome']
        X_test = test_dataset.values
        y_test_pred = classifier.predict(X_test)

        # Convert predictions to one-hot-encoded values.
        le.fit(y_test_pred)
        transform_y_test_pred = le.transform(y_test_pred)
        y_test_pred_encoded = np_utils.to_categorical(transform_y_test_pred)

        return y_test_pred_encoded
    else:
        # =================================================================================
        # Cross Validation
        # =================================================================================
        f = open(menOrWomen + '_accuracy.txt','w')
        # Cross validation
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        results = cross_val_score(classifier, X_train, y_train, cv=kfold)
        #results = cross_val_score(classifier, X_train, y_train_encoded, cv=kfold)
        print("Cross validation accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        f.write("Cross validation accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        # =================================================================================
        # Evaluate accuracy
        # =================================================================================
        # Calculate predictions
        y_train_pred = classifier.predict(X_train)

        f.write("Overall training accuracy: %5.3f%%.\n" %(100*sum(y_train==y_train_pred)/y_train.shape[0]))
        print("Overall training accuracy: %5.3f%%." %(100*sum(y_train==y_train_pred)/y_train.shape[0]))
        FE_true_positive = sum(np.bitwise_and(y_train==y_train_pred,y_train==0))
        FE_total = sum(y_train==0)
        f.write("FE Finger training accuracy: %5.3f%%.\n" %(100*FE_true_positive/FE_total))
        print("FE Finger training accuracy: %5.3f%%." %(100*FE_true_positive/FE_total))
        UE_true_positive = sum(np.bitwise_and(y_train==y_train_pred,y_train==1))
        UE_total = sum(y_train==1)
        f.write("UE Finger training accuracy: %5.3f%%.\n" %(100*UE_true_positive/UE_total))
        print("UE Finger training accuracy: %5.3f%%." %(100*UE_true_positive/UE_total))
        W_true_positive = sum(np.bitwise_and(y_train==y_train_pred,y_train==2))
        W_total = sum(y_train==2)
        f.write("W Finger training accuracy: %5.3f%%.\n" %(100*W_true_positive/W_total))
        print("W Finger training accuracy: %5.3f%%." %(100*W_true_positive/W_total))
        f.close()

        # =================================================================================
        # Plots
        # =================================================================================
        from sklearn import metrics
        from sklearn.metrics import roc_curve, confusion_matrix
        confusion matrix
        train_cnf_matrix = confusion_matrix(y_train, y_train_pred)
        print("Train Data Confusion matrix :\n%s" % train_cnf_matrix)
        plot_confusion_matrix(menOrWomen + '_confusion_train.png', train_cnf_matrix,
                             classes=['FE', 'UE', 'W'],
                             title='Training Data Confusion matrix')

        # roc metrics
        prob_y_train = classifier.predict_proba(X_train)
        plot_roc(y_train, prob_y_train, menOrWomen + '_roc_train.png')

        return y_train

def main():
    runType = 'cross-validate'
    #runType = 'predict'

    mens_predictions = runClassifier('mens_train_features.csv', 'mens_test_features.csv', 'men', runType)
    womens_predictions = runClassifier('womens_train_features.csv', 'womens_test_features.csv', 'women', runType)

    if runType == 'predict':
        outputDf = pandas.read_csv('AUS_SubmissionFormat.csv')
        results = np.concatenate((mens_predictions, womens_predictions), axis = 0)
        # Outcomes are stored in order [FE, UE, W].
        outputDf['FE'][:] = results[:, 0]
        outputDf['UE'][:] = results[:, 1]
        outputDf['W'][:] = results[:, 2]
        # Save results in csv file.
        outputDf.to_csv('results.csv', index = False)

if __name__ == "__main__":
   main()
