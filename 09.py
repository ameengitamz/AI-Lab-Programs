import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


       

data = pd.read_csv('tips.csv')
ColA = np.array(data.total_bill)
ColB = np.array(data.tip)
 

mColA = np.mat(ColA) 
mColB = np.mat(ColB)
m= np.shape(mColB)[1]

one = np.ones((1,m) ,dtype=int)



X= np.hstack((one.T,mColA.T))    

tem=np.hstack((mColA.T)) 



  
def kernel(point,xmat, k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights


 
def localWeight(point,xmat,ymat,k):
    wt = kernel(point,xmat,k)
    W = (X.T*(wt*X)).I*(X.T*(wt*ymat.T))
    return W



def localWeightRegression(xmat,ymat,k):
    
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred

 

ypred = localWeightRegression(X,mColB,2)


Xvalues=X.copy()
print(Xvalues)
Xvalues.sort(axis=0)
plt.scatter(ColA, ColB, color='blue')

plt.plot(Xvalues[:,1], ypred[X[:,1].argsort(0)], color='yellow', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show();