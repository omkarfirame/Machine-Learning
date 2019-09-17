#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
#%%
data=pd.read_csv("file:///D:/ML/Datasets/brest cancer/data.csv",sep=',')
data=data.drop(["id","Unnamed: 32"],axis=1)
X = data.drop(['diagnosis'],axis = 1)
y = data['diagnosis']


#%%
def distance(x,y):
    return np.sqrt(sum(x-y)**2)
#%%
mean1=[] 
K=[1,3,5]   
for m in K:    
    kf = KFold(n_splits=5)
    acc =[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
    
    #%%
        dd=[]    
        for i in np.arange(len(X_test)):
            dist = []
            
            for j in np.arange(len(X_train)):
                d = distance(X_test.iloc[i,:],X_train.iloc[j,:])
                dist.append(d)
                
            dd.append(dist)  
        
        matrix = np.array(dd)
        #%%
        xt = []
        for k in np.arange(len(X_test)):
              
              mat = y_train.iloc[np.argsort(matrix[k,:])[:m,]]
              tab = mat.describe()
              x = tab[2]
              xt.append(x)
    
        test1 = y_test
        tf = test1==xt
        accuracy = (sum(tf) / len(xt)) * 100
        acc.append(accuracy)
        
    mean_acc = np.mean(acc)   
    mean1.append(mean_acc)    

max_acc = np.max(mean1)
best_K = K[np.argmax(mean1)]        

print("Best value of K is ",best_K)
print("Accuracy for best K is ",max_acc)
        
  
    
        
        