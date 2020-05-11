
import numpy as np
import pandas as pd
from random import random
from random import seed
from math import exp
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



#SAMPLE DATA POINTS
sampleNum = 200

# NO OF FEATURES 
featureNum = 4

#REDUNDENT NUMBER
redundantNum = 1

#CLASSES NUMBER
classesNum = 2

#DATASET READING
X, Y = make_classification(n_samples=sampleNum, n_features=featureNum, n_redundant=redundantNum, n_classes=classesNum)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
df['label'] = Y
df.to_csv("dataset1.csv")
df=pd.read_csv('dataset1.csv',index_col=0)


#FUNCTION REQUIRED FOR TRANSFER
def transferfuntion(activater):
    return 1.0 / (1.0 + exp(-activater))


#TRANSFERING DEVRIVATIVE FUNTION
def transferDerivativefuntion(input):
    return input * (1.0 - input)


#FUNCTION FOR PREDICTIO
def predictingfunction(netw, row):
    outpt = forwardPropagatefunction(netw, row)
    return outpt.index(max(outpt))

#ACTIVATION FUNCTION
def activatefunction(wtd, input):
    activator=wtd[-1]
    for i in range(len(wtd)-1):
        activator+=wtd[i]*input[i]
    return activator




# INITIALIZING netwORK
def networkInitializingfunction(inputNum, hiddenNum, outputNum):
    netw=list()
    hiddenLayer = [{'weights':[random() for i in range(inputNum + 1)]} for i in range(hiddenNum)]
    netw.append(hiddenLayer)
    outputLayer = [{'weights':[random() for i in range(hiddenNum + 1)]} for i in range(outputNum)]
    netw.append(outputLayer)
    return netw



#BACKWARD PROPOGATION FUNCTION
def backwardPropagatefunction(netw, excpted):
    for i in reversed(range(len(netw))):
        layr = netw[i]
        err = list()
        if i != len(netw)-1:
            for j in range(len(layr)):
                error = 0.0
                for neur in netw[i + 1]:
                    error += (neur['weights'][j] * neur['delta'])
                err.append(error)
        else:
            for j in range(len(layr)):
                neur = layr[j]
                err.append(excpted[j] - neur['output'])
        for j in range(len(layr)):
            neur = layr[j]
            neur['delta'] = err[j] * transferDerivativefuntion(neur['output'])



#FORWARD PORPOGATION
def forwardPropagatefunction(netw,data):
    rwIp=data
    for i in netw:
        newRw=[]
        for j in i:
            activater=activatefunction(j['weights'], rwIp)
            j['output']=transferfuntion(activater)
            newRw.append(j['output'])
        rwIp=newRw
    return rwIp




#netwORK TRAINING FUNCTION
def trainnetworkfuntion(netw, training, l_rate, epochNum, outputNum):
    for epoch in range(epochNum):
        sum_err = 0
        for row in training:
            outpt = forwardPropagatefunction(netw, row)
            excpted = [0 for i in range(outputNum)]
            excpted[int(row[-1])] = 1
            sum_err += sum([(excpted[i]-outpt[i])**2 for i in range(len(excpted))])
            backwardPropagatefunction(netw, excpted)
            updatingWeightsfuntion(netw, row, l_rate)
        print('Loop=%d, learn_rate=%.3f, Error=%.3f' % (epoch, l_rate, sum_err))



#WEIGHTS UPDATION FUNCTION
def updatingWeightsfuntion(netw, row, l_rate):
    for i in range(len(netw)):
        inp=row[:-1]
        if i!=0:
            inp=[neur['output'] for neur in netw[i-1]]
        for neur in netw[i]:
            for j in range(len(inp)):
                neur['weights'][j]+=l_rate*neur['delta']*inp[j]
            neur['weights'][-1]+=l_rate*neur['delta']





#ARRAY IN DATASET
dataset=np.array(df[:])
dataset


#SETTING INPUT AND OUTPUT
inputNum = len(dataset[0]) - 1
outputNum = len(set([row[-1] for row in dataset]))
print(inputNum,outputNum)


#SPLIT DATASET
trainDatasetVar=dataset[:150]
testDatasetVar=dataset[150:]



#DATASET INTO netw
netw=networkInitializingfunction(inputNum,1,outputNum)
trainnetworkfuntion(netw, trainDatasetVar, 0.5, 100, outputNum)



#WEIGHTS OF netwORK
for layr in netw:
    print(layr)



#TESTING DATASET
testSet=[]
pred=[]
for row in testDatasetVar:
    prediction = predictingfunction(netw, row)
    testSet.append(row[-1])
    pred.append(prediction)
print()
print("TEST DATASET")
print("Confusion Matrix is: ",confusion_matrix(testSet,pred))
print("Accuracy is: ",accuracy_score(testSet,pred))
print("Precision is: ",precision_score(testSet, pred))
print("recall is: ",recall_score(testSet, pred))



#TRAINING DATASET
trainSet=[]
pred=[]
for row in trainDatasetVar:
    prediction = predictingfunction(netw, row)
    trainSet.append(int(row[-1]))
    pred.append(prediction)

print()
print("TRAIN DATASET")
print("Confusion Matrix is: ",confusion_matrix(trainSet,pred))
print("Accuracy is: ",accuracy_score(trainSet,pred))
print("Precision is: ",precision_score(trainSet, pred))
print("recall is: ",recall_score(trainSet, pred))
