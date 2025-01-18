import numpy as np
import matplotlib.pyplot as plt

#simoid as a function of a linear funtion, change weights for  fun.
w0=0
w1=0.5
x = np.arange(-10,10,0.1)
g_z = 1/(1+ np.exp(-(w0+w1*x)))

#plot results
plt.title('Sigmoid function for 1D with w0=3, w1=1')
plt.plot(x,g_z,color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

