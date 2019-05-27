
# coding: utf-8

# In[1]:

'''
 * @Author: 11921006 Peixin Zhang 
 * @Date: 2019-05-27
'''


# In[2]:

import numpy as np
from numpy import sin
from numpy import cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[3]:

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


# In[4]:

#test function: f(x, y) = sin(xy) + cos(xy)
def LM(x=[1.0, 1.0], miu=1e-2, epislon=1e-4):
    for k in range(1000):
        powx = np.power(x, 2)
        dot = x[0] * x[1]
        print(sin(dot) + cos(dot))
        sub = cos(dot) - sin(dot)
        add = cos(dot) + sin(dot)
        g = np.array([x[1] * sub, x[0] * sub])
        if np.linalg.norm(g) < epislon:
            break

        h = [[-powx[1] * add, sub - dot * add],
        [sub - dot * add, -powx[0] * add]]
        while not is_pos_def(h + miu * np.identity(2)):
            miu *= 4
        # (h + miu * np.identity(2)) * s = -g
        s = np.linalg.solve(h + miu * np.identity(2), -g)

        x_ = x + s
        dots = x_[0] * x_[1]
        fs = sin(dots) + cos(dots)
        f_delta = fs - (sin(dot) + cos(dot) )
        s_ = np.transpose(s)
        q_delta = g.dot(s) + 1.0 / 2 * np.matmul(np.matmul(s_, h), s)
        r = f_delta / q_delta
        if r < 0.25:
            miu *= 4
        elif r > 0.75:
            miu /= 2
        if r > 0:
            x = x + s
    dot = x[0] * x[1]
    return (x, sin(dot) + cos(dot))


# In[5]:

def main():
    optx, opty = LM()
    print(optx)
    print(opty)

    fun = lambda x, y: sin(x * y) + cos(x * y)
    fig = plt.figure()
    ax = Axes3D(fig)
    x = y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)
    ax.scatter(optx[0], optx[1], opty, c='r', s=800)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig('lm.png')
    plt.show()


# In[6]:

main()


# In[ ]:



