# -*- coding: utf-8 -*-
"""
@author: Himanshu Garg
UBID : 50292195
"""

import pickle
import gzip
from PIL import Image
import os
import numpy as np
import pandas as pd
import random
import time
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
import sklearn.metrics as skm
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
from keras.optimizers import SGD

start = time.time()

'''
softmax function is defined which takes w and x and calculates the the output y
by taking the dot product. 
'''
def softmax(w,x):
    a = np.dot(x,np.transpose(w))
    '''max of the output is subtracted so that large values don't create a 
    problem at the time of division with the sum'''
    
    num = np.exp(a-np.max(a))
    if x.ndim == 1:
        den = np.sum(num)
    else:
        den = np.sum(num,axis=1)
    softp = np.transpose(np.divide(num.T,den))
    #index = np.argmax(softp)
    #softp = softp == softp[index]
    return softp

def softLoss(y,t):
    return -np.sum(t * np.log(y + 1e-10))

'''
here the class of the output is found by picking up the maximum probabilty
'''
def getLogisticSoftPred(w,x):
    #softp = softmax(w[:,:len(w[0])-1],x)
    softp = softmax(w,np.concatenate((x,np.ones((1,x.shape[0])).T),axis=1))
    if softp.ndim == 1:
        preds = np.argmax(softp)
    else:
        preds = np.argmax(softp,axis = 1)
    return preds


'''
here the class of the output is found by picking up the maximum probabilty
'''
def getNNSoftPred(y):
    if y.ndim == 1:
        preds = np.argmax(y)
    else:
        preds = np.argmax(y,axis = 1)
    return preds 

'''
function to calculate accuracy
'''
def getAccuracy(t,y):
    d = y - t
    acc = 1.0 - (float(np.count_nonzero(d)) / len(d))
    return np.round(acc * 100,2)

'''
function to calculate the final weigts after training the classifier
'''
def getLogisticWghts(trainData,trLabels,Lamda,learningRate,epochs,batch_size):
   
    W_Now = np.zeros((10,trainData.shape[1]))
    '''random values are takin lying between 0 and 1 for the bias term'''
    b = np.random.random_sample(10)
    b.resize(1,10)
    tempOnes = np.ones((1,trainData.shape[0]))
    '''a vector of ones is appended to the last column of the data to include the 
    bias'''
    trData = np.concatenate((trainData,tempOnes.T),axis=1)
    '''the randomly generated bias values are joined with the weight matrix'''
    W_b = np.concatenate((W_Now,b.T),axis=1)
    
    for i in range(0,epochs):
        #indexes = np.random.choice(trainData.shape[0], batch_size, replace=False)
        '''random samples of size batch_size are picked from the data and 
        the weights are calculated. This is done for the number of epochs provided'''
        indexes = random.sample(range(trainData.shape[0]), batch_size)
        batch = trData[indexes,:]
        targets = trLabels[indexes,:]
        #batch = trData
        #targets = trLabels
        y = softmax(W_b,batch)
        Delta_Error    = np.dot(np.transpose(np.subtract(y,targets)),batch)
        Delta_Error    = Delta_Error/batch.shape[0]
        regularizer    = Lamda * W_b
        Delta_Error    = np.add(Delta_Error,regularizer) 
        Delta_W        = -np.dot(learningRate,Delta_Error)
        W_b            = W_b + Delta_W
    
    return W_b

def getModel(inputsize):
    
    '''
    #dropout_1 = 0.5
    dropout_2 = 0.2
    first_dense_layer_nodes  = 64
    second_dense_layer_nodes = 64
    third_dense_layer_nodes = 10
    act1 = 'relu'
    act2 = 'relu'
    act3 = 'softmax'
    optimizer = 'sgd'
    '''
    dropout_1 = 0.2
    dropout_2 = 0.2
    first_dense_layer_nodes  = 128
    second_dense_layer_nodes = 64
    third_dense_layer_nodes = 10
    act1 = 'relu'
    act2 = 'relu'
    
    '''softmax function is used for the last layer as our problem is that of 
    multi-class classification and we need the output as probabilities spanning
    10 class'''
    
    act3 = 'softmax'
    optimizer = 'sgd'
    
    
    model = Sequential()
    
    """Dense layer declares the number of neurons, weight and biases to perform
    the linear transformation on the data. It is easy to solve and are generally
    present in most neural networks. Activation function is extremely important as 
    it decides whether a neuron is to be activated or not. It is required to perform
    non-linear transformation on the data so that it can learn and solve more
    complex tasks i.e. it is able to better fit the training data. This output is 
    passed to the next layer.
    Using activation function makes the back-propagation process possible as the
    along with errors, gradients are also sent to update the weights and biases
    appropriately.
    """
    
    model.add(Dense(first_dense_layer_nodes, input_dim=inputsize))
    model.add(Activation(act1))
    
    """We need dropout to make sure that overfitting doesn't occur. i.e.
    some fraction of the nodes are dropped randomply so that machine learning 
    algorithm while training with a large number of epochs doesn't get biased,
    and is able to learn correctly.
    """
    
    model.add(Dropout(dropout_1))       
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation(act2))
    model.add(Dropout(dropout_2))    
    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation(act3))
    model.summary()
    
    """
    Stochastic gradient descent optimizer is used and the learning varied to get
    the best accuracy.
    """
    lr = 0.25
    sgd = SGD(lr=lr)
    
    """categorical_crossentropy loss function is used when we have multi-class
    classification problem. In our case we have 10 classes : 0 to 9. Our aim here is to 
    minimize this loss function to improve the accuracy.
    """
    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    print("Input Dimension: ",inputsize)
    print("First Dense Layer Nodes: ",first_dense_layer_nodes)
    print("First Layer activation: ",act1)
    print("First Layer dropout: ",dropout_1)
    print("Second Dense Layer Nodes: ",second_dense_layer_nodes)
    print("Second Layer activation: ",act2)
    print("Second Layer dropout: ",dropout_2)
    print("Third Dense Layer Nodes: ",third_dense_layer_nodes)
    print("Third Layer activation: ",act3)
    print("Optimizer: ",optimizer)
    print("Learning rate: ",lr)
    return model

def loadmnistdata():
    filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    mnistTrData, mnistValData, mnistTestData = pickle.load(f, encoding='latin1')
    f.close()
    return mnistTrData, mnistValData, mnistTestData

def loaduspsdata():
    USPSMat  = []
    USPSTar  = []
    curPath  = 'USPSdata/Numerals'
    savedImg = []
    
    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                savedImg = img
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)
    USPSMat = np.array(USPSMat)
    USPSTar = np.array(USPSTar)
    return USPSMat,USPSTar

def runLogisticR(mnistTrData,mnistValData,mnistTestData,mnistTrLabels,Lamda,learningRate,epochs,batch_size):
    #La = [10,50,100,500,1000,10000,20000]
    #mnacc = []
    #usacc = []
    #for epochs in La:
    W = getLogisticWghts(mnistTrData[0],mnistTrLabels,Lamda,learningRate,epochs,batch_size)

    print("\nLamda: ",Lamda)
    print("learning Rate: ",learningRate)
    print("Batch size: ",batch_size)
    print("Epochs: ",epochs)
    mnvalpredictions = getLogisticSoftPred(W,mnistValData[0])
    mnvalAcc = getAccuracy(mnvalpredictions,mnistValData[1])
    print("MNIST data Validation Accuracy:",mnvalAcc)
    mntestpredictions = getLogisticSoftPred(W,mnistTestData[0])
    mntestAcc = getAccuracy(mntestpredictions,mnistTestData[1])
    print("MNIST data Testing Accuracy:",mntestAcc)
    #mnacc.append(mntestAcc)
    mnconfmatrix = skm.confusion_matrix(mnistTestData[1],mntestpredictions,labels=[0,1,2,3,4,5,6,7,8,9])
    print("Confusion Matrix for MNIST dataset:\n",np.array(mnconfmatrix))
    
    uspredictions = getLogisticSoftPred(W,USPSMat)
    usAcc = getAccuracy(uspredictions,USPSTar)
    print("\nUSPS data Accuracy:",usAcc)
    #usacc.append(usAcc)
    mnconfmatrix = skm.confusion_matrix(USPSTar,uspredictions,labels=[0,1,2,3,4,5,6,7,8,9])
    print("Confusion Matrix for USPS dataset:\n",np.array(mnconfmatrix))
    '''
    plt.plot(La, mnacc) 
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy for MNIST Test') 
    plt.title('Plot of Accuracy vs Epochs for MNIST Test set') 
    plt.show()
    plt.plot(La, usacc) 
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy for USPS Test') 
    plt.title('Plot of Accuracy vs Epochs for USPS Test set') 
    plt.show()
    '''
    return mntestpredictions, uspredictions

def runPerceptron(mnistTrData,mnistTrLabels,mnistValData,mnistValLabels,mnistTestData,mnistTestLabels):
    #La = [0.005,0.01,0.05,0.1,0.5,1,2]
    #mnacc = []
    #usacc = []
    #for lr in La:
    model = getModel(mnistTrData[0].shape[1])
    num_epochs = 300
    print("Number of epochs: ",num_epochs)
    model_batch_size = 128
    early_patience = 100
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
    history = model.fit(mnistTrData[0]
                    , mnistTrLabels
                    , validation_data=(mnistValData[0],mnistValLabels)
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [earlystopping_cb]
                    , verbose = 0
                   )
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10,15))
    plt.show()
    score = model.evaluate(mnistTestData[0], mnistTestLabels, batch_size=model_batch_size)
    mnpredictedTargets = model.predict(mnistTestData[0]) 
    mnpredictedTargets = getNNSoftPred(mnpredictedTargets)
    print("\nNeural Network accuracy for MNIST training set: ",history.history['acc'][len(history.history['acc'])-1] * 100)
    #print("Perceptron loss for MNIST training set: ",history.history['loss'][len(history.history['loss'])-1])
    print("Perceptron accuracy for MNIST validation set: ",history.history['val_acc'][len(history.history['val_acc'])-1] * 100)
    #print("Perceptron loss for MNIST validation set: ",history.history['val_loss'][len(history.history['val_loss'])-1])
    #print("Perceptron accuracy for MNIST Test set:",getAccuracy(mnpredictedTargets,mnistTestData[1]))
    print("Perceptron accuracy for MNIST Test set:",np.round(score[1]*100,2))
    #mnacc.append(np.round(score[1]*100,2))
    #print("Perceptron loss for MNIST Test set:",score[0])
    confmatrix = skm.confusion_matrix(mnistTestData[1],mnpredictedTargets,labels=[0,1,2,3,4,5,6,7,8,9])
    print("\nConfusion Matrix for MNIST dataset:\n",np.array(confmatrix))
    score = model.evaluate(USPSMat, uspsTestLabels, batch_size=model_batch_size)
    uspredictedTargets = model.predict(USPSMat) 
    uspredictedTargets = getNNSoftPred(uspredictedTargets)
    #print("\nPerceptron accuracy for USPS dataset:",getAccuracy(uspredictedTargets,USPSTar))
    print("\nPerceptron accuracy for USPS dataset:",np.round(score[1]*100,2))
    #usacc.append(np.round(score[1]*100,2))
    #print("Perceptron loss for USPS dataset:",score[0])
    confmatrix = skm.confusion_matrix(USPSTar,uspredictedTargets,labels=[0,1,2,3,4,5,6,7,8,9])
    print("\nConfusion Matrix for USPS dataset:\n",np.array(confmatrix))
    '''
    plt.plot(La, mnacc) 
    plt.xlabel('Learning Rate') 
    plt.ylabel('Accuracy for MNIST Test') 
    plt.title('Plot of Accuracy vs Learning Rate for MNIST Test set') 
    plt.show()
    plt.plot(La, usacc) 
    plt.xlabel('Learning Rate') 
    plt.ylabel('Accuracy for USPS Test') 
    plt.title('Plot of Accuracy vs Learning Rate for USPS Test set') 
    plt.show()
    '''
    return mnpredictedTargets,uspredictedTargets
    
def runSVM(mnistTrData,mnistValData,mnistTestData):
    svc = SVC( C=2, kernel='rbf', gamma = 0.05, decision_function_shape='ovo')
    """
    SVM classifier is taken here and performance measured by varying the kernel,
    and the gamma
    """
    #svc = SVC( kernel='rbf', gamma = 0.05)
    #svc = SVC( kernel='rbf')
    #svc = SVC( kernel='linear' )
    svc.fit(mnistTrData[0], mnistTrData[1])
    
    mnvalpredictions = svc.predict(mnistValData[0])
    mntestpredictions = svc.predict(mnistTestData[0])
    print("\nSVM accuracy for MNIST validation set:",getAccuracy(mnvalpredictions,mnistValData[1]))
    print("SVM accuracy for MNIST test set:",getAccuracy(mntestpredictions,mnistTestData[1]))
    confmatrix = skm.confusion_matrix(mnistTestData[1],mntestpredictions,labels=[0,1,2,3,4,5,6,7,8,9])
    print("\nConfusion Matrix for MNIST dataset:\n",np.array(confmatrix))
    
    ustestpredictions = svc.predict(USPSMat)
    #testscore = svc.score(testpredictions,USPSTar)
    print("\nSVM accuracy for USPS test set:",getAccuracy(ustestpredictions,USPSTar))
    confmatrix = skm.confusion_matrix(USPSTar,ustestpredictions,labels=[0,1,2,3,4,5,6,7,8,9])
    print("\nConfusion Matrix for USPS dataset:\n",np.array(confmatrix))
    
    return mntestpredictions,ustestpredictions

def runRandomForest(mnistTrData,mnistValData,mnistTestData):
    #rf = RandomForestClassifier(n_estimators = 50,max_depth = 30)
    #rf = RandomForestClassifier(n_estimators = 200)
    trees = 200
    """
    Random forest  classifier is taken here and performance measured by varying the
    the number of trees given by the variable n_estimators
    """
    rf = RandomForestClassifier(n_estimators = trees)
    rf.fit(mnistTrData[0], mnistTrData[1])
    mnvalpredictions = rf.predict(mnistValData[0])
    mntestpredictions = rf.predict(mnistTestData[0])
    print("\nNumber of Trees: ",trees)
    #print("Depth: ","Default")
    print("\nRandom Forest accuracy for MNIST validation set:",getAccuracy(mnvalpredictions,mnistValData[1]))
    print("Random Forest accuracy for MNIST test set:",getAccuracy(mntestpredictions,mnistTestData[1]))
    confmatrix = skm.confusion_matrix(mnistTestData[1],mntestpredictions,labels=[0,1,2,3,4,5,6,7,8,9])
    print("\nConfusion Matrix for MNIST dataset:\n",np.array(confmatrix))
    
    ustestpredictions = rf.predict(USPSMat)
    print("\nRandom Forest accuracy for USPS test set:",getAccuracy(ustestpredictions,USPSTar))
    confmatrix = skm.confusion_matrix(USPSTar,ustestpredictions,labels=[0,1,2,3,4,5,6,7,8,9])
    print("\nConfusion Matrix for USPS dataset:\n",np.array(confmatrix))
    
    return mntestpredictions,ustestpredictions

"""
here the predictions from all the four classifiers are combined together and 
Majority Voting is used to get the final predictions and final accuracy calculated
"""
def ensemble(predArr):
    fpreds = []
    for l1,l2,l3,l4 in zip(predArr[0],predArr[1],predArr[2],predArr[3]):
        temp = [l1,l2,l3,l4]
        unique, counts = np.unique(temp, return_counts=True)
        ind = np.argmax(counts)
        label = unique[ind]
        fpreds.append(label)
    return fpreds
        


    
'''############################### START ###################################'''

mnistTrData, mnistValData, mnistTestData = loadmnistdata()
print("MNIST training feature shape: ",mnistTrData[0].shape)
print("MNIST training label shape: ",mnistTrData[1].shape)
print("MNIST validation feature shape: ",mnistValData[0].shape)
print("MNIST validation label shape: ",mnistValData[1].shape)
print("MNIST testing feature shape: ",mnistTestData[0].shape)
print("MNIST testing label shape: ",mnistTestData[1].shape)
mnistTrLabels = np_utils.to_categorical(np.array(mnistTrData[1]),10)
mnistValLabels = np_utils.to_categorical(np.array(mnistValData[1]),10)
mnistTestLabels = np_utils.to_categorical(np.array(mnistTestData[1]),10)

USPSMat, USPSTar = loaduspsdata()
print("USPS feature shape: ",USPSMat.shape)
print("USPS label shape: ",USPSTar.shape)
uspsTestLabels = np_utils.to_categorical(np.array(USPSTar),10)

Lamda = 0.001
learningRate = 0.09
batch_size = 100
epochs = 12000
mnensembPreds = []
usensembPreds = []


print("\n-----------------LOGISTIC REGRESSION USING SOFTMAX-------------------")
lrsMNISTpreds, lrsUSPSpreds = runLogisticR(mnistTrData,mnistValData,mnistTestData,mnistTrLabels,Lamda,learningRate,epochs,batch_size)
mnensembPreds.append(lrsMNISTpreds)
usensembPreds.append(lrsUSPSpreds)
print("-----------------------------------------------------------------------")


print("\n--------------------PERCEPTRON NEURAL NETWORK------------------------")
percMNISTpreds,percUSPSpreds = runPerceptron(mnistTrData,mnistTrLabels,mnistValData,mnistValLabels,mnistTestData,mnistTestLabels)
mnensembPreds.append(percMNISTpreds)
usensembPreds.append(percUSPSpreds)
print("-----------------------------------------------------------------------")



print("\n----------------------------- SVM -----------------------------------")
svmMNISTpreds,svmUSPSpreds = runSVM(mnistTrData,mnistValData,mnistTestData)
mnensembPreds.append(svmMNISTpreds)
usensembPreds.append(svmUSPSpreds)
print("-----------------------------------------------------------------------")



print("\n-------------------------RANDOM FOREST-------------------------------")
rfMNISTpreds,rfUSPSpreds = runRandomForest(mnistTrData,mnistValData,mnistTestData)
mnensembPreds.append(rfMNISTpreds)
usensembPreds.append(rfUSPSpreds)
print("-----------------------------------------------------------------------")



print("\n---------------------------- ENSEMBLE -------------------------------")
fmnpreds = ensemble(np.array(mnensembPreds))
print("\nFinal accuracy for MNIST dataset after Ensemble: ",getAccuracy(fmnpreds,mnistTestData[1]))
confmatrix = skm.confusion_matrix(mnistTestData[1],fmnpreds,labels=[0,1,2,3,4,5,6,7,8,9])
print("\nConfusion Matrix for MNIST dataset after Ensemble:\n",np.array(confmatrix))

fuspreds = ensemble(np.array(usensembPreds))
print("Final accuracy for USPS dataset after Ensemble: ",getAccuracy(fuspreds,USPSTar))
confmatrix = skm.confusion_matrix(USPSTar,fuspreds,labels=[0,1,2,3,4,5,6,7,8,9])
print("\nConfusion Matrix for USPS dataset after Ensemble:\n",np.array(confmatrix))
print("-----------------------------------------------------------------------")

end = time.time()
print("Elapsed time: ",end-start)