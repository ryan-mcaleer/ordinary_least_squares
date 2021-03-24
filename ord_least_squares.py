#!/usr/bin/env python
# coding: utf-8

# In[25]:


import matplotlib.pyplot as plt
import numpy as np


# In[26]:


def least_squares(mat, arr, m, n):
    #matrix for A.T * A
    AT_A = np.zeros((n,n))
    #Right hand side vector
    b = np.zeros((1,n)).T

    #Solution Vector
    x_hat = np.zeros((1,n)).T
    
    #Calculate A.T * A
    for i in range(m):
        b[0,0] += mat[i,0] * arr[i]
        b[1,0] += mat[i,1] * arr[i]
        for j in range(n):
            AT_A[0,j] += mat[i,0] * mat[i,j]
            AT_A[1,j] += mat[i,1] * mat[i,j]

    #Perform Elimination on A.T * A & right side b        
    row_multiple = AT_A[1,0]/AT_A[0,0]
    AT_A[1] -= (row_multiple)*AT_A[0]
    b[1,0] -= row_multiple*b[0,0]

    
    #Solve for x-hat
    x_hat[1,0] = b[1,0] / AT_A[1,1]
    x_hat[0,0] = (b[0,0] - AT_A[0,1]*(b[1,0] / AT_A[1,1])) / AT_A[0,0]
    return x_hat


# In[27]:


mat = np.array([
    [1.,1.],
    [1.,2.],
    [1.,3.],
    [1.,4.],
])

arr = np.array([
                [1., 2., 3., 3.]
]).T

x_hat = least_squares(mat, arr, 4, 2)

#set up grid and plot points
plt.plot(mat[:,1], arr[:,0], "o")
plt.axis([0,6,0,6])

#create x and f(x) to be plotted
x = np.linspace(0,6,100)
y = x_hat[0,0] + x_hat[1,0] * x

#plot regression line
plt.plot(x,y)


# In[ ]:





# In[ ]:





# In[ ]:




