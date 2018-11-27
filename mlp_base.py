# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:09:27 2018

@author: gaoming
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from neural_net import TwoLayerNet
from sklearn import preprocessing


data=pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
X=data.iloc[:,:20].astype('category')
for i in range(X.shape[1]):
    X.iloc[:,i]=X.iloc[:,i].cat.codes
    
y=data.iloc[:,-1].astype('category').cat.codes

input_size=X.shape[1]
hidden_size=20
num_classes=2

#np.random.seed(0)
traind, testd, trainy, testy = train_test_split(X, y, train_size=0.75)

traindd=traind.values; testdd=testd.values
trainyy=trainy.values; testyy=testy.values

data_scaler = preprocessing.MinMaxScaler()
traindd = data_scaler.fit_transform(traindd) ; testdd = data_scaler.fit_transform(testdd)

#display
traindd[:5,]
trainyy[:5,]
#np.random.seed(0)
net=TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)
stats = net.train(traindd, trainyy, 
            learning_rate=1e-1, reg=5e-6,
            num_iters=180, batch_size=40, verbose=False)

#print('Final training loss: ', stats['loss_history'][-1])
print(np.mean(net.predict(testdd)==testyy))
# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.savefig('trainloss.png')
plt.show()
