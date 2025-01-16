## Linear Regression/ Curve Fitting - Not Normalized

import numpy as np
import matplotlib.pyplot as plt

#Test Data
x = np.array([150.0, 250.0, 350.0, 450.0, 550.0, 650.0, 750.0 ])
y = np.array([63.6, 59.7, 46.6, 41.2, 27.5, 19.8, 12.3])

#Gradient descent method Parameters
w0=10          # Initial Guess
w1=1       # Initial Guess
lr=0.0000002   # Learning
epsilon=0.001

for i in range(100):
  #plot values of our guess
  yguess = w0 + w1*x
  error=sum((y-(w0 + w1*x))**2)/2
  #print(error,w0,w1)
  plt.plot(x,yguess, color='grey')

  #Gradients
  #Weights
  G0 = sum(yguess - y)  # Gradient with respect to w0
  G1 = sum(x * (yguess - y))  # Gradient with respect to w1
  w0 = w0 - lr * G0  # Update w0
  w1 = w1 - lr * G1  # Update w1

# plot results
plt.title('Tempering Example')
plt.xlabel('Tempering Temperation in degree C')
plt.ylabel('Rockwell C hardness')
plt.plot(x,yguess, color='blue')
plt.scatter(x,y)
plt.grid()
plt.show()