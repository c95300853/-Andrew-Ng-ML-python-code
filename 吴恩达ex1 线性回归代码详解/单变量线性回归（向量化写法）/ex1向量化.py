import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def read_data(path):
    data1=pd.read_table(path,header=None)
    print(data1.shape)
    data=pd.read_table(path,header=None,sep=",")
    print(data)
    x=data.iloc[:,0]
    y=data.iloc[:,1]
    print(type(x),type(y))
    m=len(x)
    x=np.array(x)
    y=np.array(y)
    # x,y=x.T,y.T
    return x,y,m

def init_data():
    # w=random.random()
    # b=random.random()
    w=1
    b=1
    learningRate=0.001
    return w,b,learningRate

def gradientDescent(w,b,y,learningRate,m,x):
    w=w-learningRate*(((((w*x+b)-y)*x).sum())/m)
    b=b-learningRate*(((((w*x+b)-y)).sum())/m)
    cost=((((w * x + b) - y)*((w * x + b) - y)).sum())/(2*m)
    return w,b,cost

path="ex1data1.txt"
arr_cost=[]
x,y,m=read_data(path)
w,b,learningRate=init_data()
for i in range(50000):
    w,b,cost=gradientDescent(w,b,y,learningRate,m,x)
    arr_cost.append(cost)
plt.plot(arr_cost)
plt.show()