from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
iris=datasets.load_iris()
X=iris.data[:,[2,3]]

X1=iris.data[0:100,[2,3]]
y=iris.target[0:100]
Xt=iris.data[35:75,[2,3]]
yt=iris.target[35:75]
X1s=X1.copy()
Xts=Xt.copy()
X1s=(X1-X1.mean())/X1.std()
Xts=(Xt-Xt.mean())/Xt.std()
X01=np.ones(100)
X02=np.ones(40)
thetaz=0
theta0=0
theta1=0
alpha=0.1
errorgrad0=0.0
errorgradz=0.0
errorgrad1=0.0

i=0
for i in range(20):
    j=0
    for j in range(100):
        
        g=thetaz*1+theta0*X1s[j,0]+theta1*X1s[j,1]

        errorgradz+=g-y[j]
        errorgrad0+=(g-y[j])*X1s[j,0]
        errorgrad1+=(g-y[j])*X1s[j,1]
        
    thetaz-=alpha*errorgradz/100
    theta0-=alpha*errorgrad0/100
    theta1-=alpha*errorgrad1/100
    h=thetaz*X01+theta0*X1s[:,0]+theta1*X1s[:,1]
    hn=thetaz*X02+theta0*Xts[:,0]+theta1*Xts[:,1]
    h=np.where(h>=0,1,0)
    
    
thetaz=0.024574177124360802
theta0=0.6522284580068151
theta1=0.23745197723155068
h=thetaz*X01+theta0*X1s[:,0]+theta1*X1s[:,1]
hn=thetaz*X02+theta0*Xts[:,0]+theta1*Xts[:,1]
h=np.where(h>=0,1,0)
hn=np.where(hn>=0,1,0)

i=0
x0=np.array([])
x1=np.array([])
x2=np.array([])
x3=np.array([])

while(i<h.size):
    if(h[i]==0):
        x0=np.append(x0,X1s[i,0])
        x1=np.append(x1,X1s[i,1])
    else:
        x2=np.append(x2,X1s[i,0])
        x3=np.append(x3,X1s[i,1])
    i+=1  
plt.scatter(x0,x1,c="red",label="0...setosa")
plt.scatter(x2,x3,c="blue",label="1...versicolor")
plt.legend()
plt.show()
    
