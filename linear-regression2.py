
import numpy as np
import matplotlib.pyplot as plt

#Create Test Data
x = np.arange(0,np.pi/3,0.1)
y = np.sin(2*np.pi*x)
[y_noise] = y+ np.random.randn(1,len(y))/5


# For Plotting
xplot = np.arange(0,np.pi/3,0.01)

#Gradient descent method Parameters
theata0=  0  # 0.31
theata1=  5  # 7.99
theata2=  -21   #-25.43
theata3=  14 # 17.37
lr=0.02        # Learning Rate


for i in range(100):
  #plot values of our guess
  yplot=theata0 + theata1*xplot + theata2*xplot**2+ theata3*xplot**3

  #Calculate guess for Gradient descent
  yguess = theata0 + theata1*x + theata2*x**2+ theata3*x**3
  error=sum((y_noise-yguess)**2)/2
  #print(error,theata0,theata1,theata2,theata3)
  plt.plot(xplot,yplot, color='grey')

  #Gradients
  G0 = sum(yguess - y_noise)        # Gradient with respect to w0
  G1 = sum(x * (yguess - y_noise))  # Gradient with respect to w1
  G2 = sum(x**2 * (yguess - y_noise))  # Gradient with respect to w2
  G3 = sum(x**3 * (yguess - y_noise))  # Gradient with respect to w2
  #Weights
  theata0 = theata0 - lr * G0  # Update w0
  theata1 = theata1 - lr * G1  # Update w1
  theata2 = theata2 - lr * G2  # Update w2
  theata3 = theata3 - lr * G3  # Update w3


# Plot Results
plt.title('Tempering Example')
plt.xlabel('Tempering Temperation in degree C')
plt.ylabel('Rockwell C hardness')
plt.plot(xplot,yplot, color='blue')
plt.scatter(x,y_noise)
plt.grid()
plt.show()

