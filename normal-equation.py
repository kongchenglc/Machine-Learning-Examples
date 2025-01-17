
import numpy as np
# Using Normal Equations All Test Data

#Features in this case theat0, theta1 paramaters and theta 1 is the bias and =1
X = np.array(([1, 150.0],
              [1, 250.0],
              [1, 350.0],
              [1, 450.0],
              [1, 550.0],
              [1, 650.0],
              [1, 750.0]))
Y = np.array([63.6, 59.7, 46.6, 41.2, 27.5, 19.8, 12.3])

# X transpose X
XtX=np.matmul(np.transpose(X),X)
print('X transpose X :\n',XtX)
XTXinv=np.linalg.inv(XtX)
print('\n Inverse(X transpose X) :\n',XTXinv)
XtXinvXT= np.matmul(XTXinv,np.transpose(X))
print('\n Inverse(X transpose X)*X transpose :\n', XtXinvXT)
theta=np.matmul(XtXinvXT,np.array(Y))
print('\n and...Theta, the intercept and slope are...\n',theta)