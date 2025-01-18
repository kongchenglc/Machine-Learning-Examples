import numpy as np
import matplotlib.pyplot as plt

#Decision Boundary, go ahead and play with the weights, wo and w1
w0=5
w1=3
x = np.arange(-10,10,0.1)
g_z = 1/(1+ np.exp(-(w0+w1*x)))
g_z_neg = np.exp(-(w0+w1*x))/(1+ np.exp(-(w0+w1*x)))
[index]=np.where(g_z<0.5) #find the index where y==0.5
print(x[max(index)])

#plot results
plt.title('Sigmoid function Decision Boundary')
plt.plot(x,g_z,color='green')
plt.plot(x,g_z_neg,color='red')
plt.plot(x[max(index)]*np.ones(len(x)),g_z_neg,color='black',linestyle=':')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

