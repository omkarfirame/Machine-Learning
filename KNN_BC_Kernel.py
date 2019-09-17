# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:03:05 2019

@author: Omkar
"""

import numpy as np
import pandas as pd


data=pd.read_csv("file:///D:/ML/Datasets/brest cancer/data.csv",sep=',')
data=data.drop(["id","Unnamed: 32"],axis=1)

test = data[455:569]
train = data[:454]
dd = []

def poly(x1,x2):
    return (1 + np.dot(x1,x2))

def linear(x1,x2):
    return np.dot(x1,x2)



def distance(x,y,Z):
    return np.sqrt((x)**2+(y)**2-2*(Z)**2)

for i in np.arange(len(test)):
    dist = []
    
    for j in np.arange(len(train)):
        Q = test.drop(columns = ["diagnosis"]).iloc[i,:]
        W = np.transpose(Q)
        E = train.drop(columns = ["diagnosis"]).iloc[j,:]
        R = np.transpose(E)
        a1 = (1+np.dot(Q,W))**2
        a2 = (1+np.dot(E,R))**2
        a3 = (1+np.dot(Q,R))**2
        d = distance(a1,a2,a3)
        dist.append(d)
        
    dd.append(dist)  

matrix = np.array(dd)

xt = []
       

for k in range(114):
      mat = train.diagnosis[np.argsort(matrix[k,:])[:3]]
      tab = mat.describe()
      x = tab[2]
      xt.append(x)
      
test1 = test.diagnosis
tf = test1==xt
accuracy = (sum(tf) / len(xt)) * 100

print(accuracy)
 
        
  
    
        
        