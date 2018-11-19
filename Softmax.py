# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 23:14:28 2018

@author: Joby
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    """Compute softmax values of x"""
    x=np.exp(x)
    x = x/sum(x)
    return x

scores = np.array([3.0, 2.0, 0.1])
print(softmax(scores))
print(softmax(scores*10))
print(softmax(scores/10))

x = np.arange(-2.0, 6.0, 0.1)

scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
print(softmax([[-2,-1,0],[1,1,1],[0.2,0.2,0.2]]))
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
trial = 1E9
print(trial)
error = 1E-6
for i in range(1000000):
    trial += error
trial=trial-1E9
print(trial)
