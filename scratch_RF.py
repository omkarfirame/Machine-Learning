# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:04:50 2019

@author: Omkar
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV,train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,accuracy_score,make_scorer
from sklearn.ensemble import RandomForestClassifier

X = load_breast_cancer().data
y = load_breast_cancer().target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
acc2=[]
for i in np.arange(2,10,2):
    acc=[]
    for j in np.arange(50,500,50):
        model = RandomForestClassifier(n_estimators=j,max_features=i)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        score = make_scorer(accuracy_score)
        cv = cross_val_score(estimator=model,X=X,y=y,cv=5,scoring=score)
        m_acc = np.mean(cv)
        acc.append(m_acc)
    acc1 = np.max(acc)
    nt = np.argmax(acc)
    ntree = np.arange(50,500,50)[nt]
    acc2.append(acc1)
    accuracy = np.max(acc2)
    mt = np.argmax(acc2)
    mtry = np.arange(2,10,2)[mt]
ac=[]
for j in np.arange(50,500,50):
    model1 = RandomForestClassifier(n_estimators=j)
    model1.fit(X_train,y_train)
    y_pred1 = model1.predict(X_test)
    score1 = make_scorer(accuracy_score)
    cv1 = cross_val_score(estimator=model1,X=X,y=y,cv=5,scoring=score1)
    m_acc1 = np.mean(cv1)
    ac.append(m_acc1)
ac1 = np.max(ac)
nt1 = np.argmax(ac)
ntree1 = np.arange(50,500,50)[nt1]

ac2=[]
for i in np.arange(2,10,2):
    model2 = RandomForestClassifier(max_features=i)
    model2.fit(X_train,y_train)
    y_pred2 = model2.predict(X_test)
    score2 = make_scorer(accuracy_score)
    cv2 = cross_val_score(estimator=model2,X=X,y=y,cv=5,scoring=score2)
    m_acc2 = np.mean(cv2)
    ac2.append(m_acc2)
ac3 = np.max(ac2)
mt1 = np.argmax(ac2)
mtry1 = np.arange(2,10,2)[mt1]
print("For default Mtry and Varying Ntree Best Accuracy is %f with Best Ntree is %d"%(ac1,ntree1))
print("For default Ntree and Varying Mtry Best Accuracy is %f with Best Mtry is %d"%(ac3,mtry1))
print("Best Mtry is %d, Best Ntree is %d with Best Accuracy %f"%(mtry,ntree,accuracy))
