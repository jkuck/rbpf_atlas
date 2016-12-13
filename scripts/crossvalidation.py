import numpy as np
import json
import sys
tracks = json.load(open("../data/groundtruth_tracks.json"))
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import theano
theano.config.openmp = True

def buildLookbackDataset(tracks, lookback=3, skip=None):
    lookback +=1
    dataset = []
    for key, track in tracks.items():
        if skip:
            filename = "%04d.txt" % skip
            if filename in key:
                continue
        for i, pos in enumerate(track):
            if i < lookback:
                continue
            dataset.append(np.array(track[i-lookback:i]).ravel())
    return dataset

def buildVarDataset(tracks, predTracks, variance, lookback=3, skip=None):
    lookback += 1
    dataset = []
    for key, track in tracks.items():
    	if skip:
            filename = "%04d.txt" % skip
            if filename in key:
                continue
        for i, pos in enumerate(track):
            if i < lookback:
                continue
            gt = np.array(track[i-lookback:i-1]).ravel()
            pred = np.array(predTracks[key][i])
            var = np.array(variance[key][i])
            row = np.concatenate((gt, pred, var))
            dataset.append(row)
    return dataset

def mle_var(tracks, predTracks, skip=None):
    variances = {}
    x = []
    y = []
    for (key, track) in tracks.items():
        if skip:
            filename= "%04d.txt" % skip
            if filename in key:
                continue
        variances[key] = []
        for j, point in enumerate(track):
            x = ((point[0] - predictedTracks[key][j][0])**2)
            y = ((point[1] - predictedTracks[key][j][1])**2)
            variances[key].append((x,y))
    return variances

def rebuildTracks(tracks, model, skip=None):
    predTracks = {}
    for track in tracks.items():
        if skip:
            filename = "%04dtxt" % skip
            if filename in track[0]:
                continue
        predTracks[track[0]] = []
        t = predTracks[track[0]]
        for i, pos in enumerate(track[1]):
            if i < 3:
                t.append(pos)
            else:
                # scale and resize
                p1 = track[1][i-3]
                p2 = track[1][i-2]
                p3 = track[1][i-1]
                p = scaler.transform(np.array(p1+p2+p3+p3).reshape(1,-1))[0][0:6].reshape(1,3,2)
                # predict
                pred = model.predict(p).reshape(1,2).tolist()
                pred = scaler.inverse_transform(np.array(pred[0]+pred[0]+pred[0]+pred[0])\
                                                .reshape(1,-1))[0][0:2]
                t.append(pred)
    return predTracks
      
i = int(sys.argv[1])
   
print "Cross validation for model %d" % i
dataset = buildLookbackDataset(tracks,skip=i) 
dataset = np.array(dataset)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

xtrain = np.matrix(train[:,0:6])
ytrain = np.matrix(train[:,6:])
xtest = np.matrix(test[:,0:6])
ytest = np.matrix(test[:,6:])

# reshape input to be [samples, time steps, features]
xtrain = np.reshape(np.array(xtrain), (xtrain.shape[0], 3, 2))
xtest = np.reshape(np.array(xtest), (xtest.shape[0], 3, 2))

model = Sequential()
model.add(LSTM(32, input_shape=(3,2)))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(xtrain, ytrain, nb_epoch=10, batch_size=1, verbose=2)
model.save_weights('/lfs/local/0/daniter/autocars/cv-mu-weights%d.h5' % i)

trainPredict = model.predict(xtrain)
testPredict = model.predict(xtest)

scaledPredictTrain = scaler.inverse_transform(np.concatenate((ytrain,trainPredict,ytrain,trainPredict), axis=1))
scaledPredictTest = scaler.inverse_transform(np.concatenate((ytest,testPredict,ytest,testPredict), axis=1))
scaledyTrain = scaler.inverse_transform(np.concatenate((ytrain,trainPredict,ytrain,ytrain), axis=1))
scaledyTest = scaler.inverse_transform(np.concatenate((ytest,testPredict,ytest,ytest), axis=1))

trainScore = math.sqrt(mean_squared_error(scaledyTrain[:,6:], scaledPredictTrain[:,6:]))
print('Train Score: %.7f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(scaledyTest[:,6:], scaledPredictTest[:,6:]))
print('Test Score: %.7f RMSE' % (testScore))

predictedTracks = rebuildTracks(tracks,model)
vs = mle_var(tracks, predictedTracks)
dataset = buildVarDataset(tracks, predictedTracks, vs, skip=i) 
dataset = np.array(dataset)

dataset[:,8] = dataset[:,8].clip(0,100)
dataset[:,9] = dataset[:,9].clip(0,5)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into test and train
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

xtrain = np.matrix(train[:,0:8])
ytrain = np.matrix(train[:,8:])
xtest = np.matrix(test[:,0:8])
ytest = np.matrix(test[:,8:])

# reshape input to be [samples, time steps, features]
xtrain = np.reshape(np.array(xtrain), (xtrain.shape[0], 4, 2))
xtest = np.reshape(np.array(xtest), (xtest.shape[0], 4, 2))

varmodel = Sequential()
varmodel.add(LSTM(32, input_shape=(4,2)))
varmodel.add(Dense(2))
varmodel.compile(loss='mean_squared_error', optimizer='adam')
varmodel.fit(xtrain, ytrain, nb_epoch=10, batch_size=1, verbose=2)
varmodel.save_weights('/lfs/local/0/daniter/autocars/cv-cov-weights%d.h5' % i)

trainPredict = varmodel.predict(xtrain)
testPredict = varmodel.predict(xtest)

scaledYTrain = scaler.inverse_transform(np.concatenate((xtrain.reshape((xtrain.shape[0],8)),ytrain), axis=1))
scaledYTest = scaler.inverse_transform(np.concatenate((xtest.reshape((xtest.shape[0],8)),ytest), axis=1))
scaledPredictTrain = scaler.inverse_transform(np.concatenate((xtrain.reshape((xtrain.shape[0],8)),trainPredict), axis=1))
scaledPredictTest = scaler.inverse_transform(np.concatenate((xtest.reshape((xtest.shape[0],8)),testPredict), axis=1))

trainScore = math.sqrt(mean_squared_error(scaledPredictTrain[:,8:], scaledYTrain[:,8:]))
print('Train Score: %.7f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(scaledPredictTest[:,8:], scaledYTest[:,8:]))
print('Test Score: %.7f RMSE' % (testScore))





