# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:38:52 2020

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
 
x = np.linspace(-1, 6, 1000)
 
y12 = -x + 5
y13 = np.array([3]*1000)
y23 = x - 1
y_T = np.array([6]*1000)
y_B = np.array([-2]*1000)

ax = plt.gca()

gamma1 = plt.fill_between(x, y_B, np.append(y13[x<2], y12[x>2]), facecolor = "green")
gamma2 = plt.fill_between(x[x>3], y12[x>3], y23[x>3], facecolor = "blue")
gamma3 = plt.fill_between(x, np.append(y13[x<4], y23[x>4]), y_T, facecolor = "red")
gamma4 = plt.fill_between(x[np.where((x>2)&(x<4))], np.append(y12[np.where((x>2)&(x<=3))], y23[np.where((x>3)&(x<4))]), y13[np.where((x>2)&(x<4))], facecolor = "white")

plt.legend(handles=[gamma1, gamma2, gamma3, gamma4], labels=["gamma1", "gamma2", "gamma3", 'indeterminate'])
plt.plot(x, y12, c = "g")
plt.plot(x, y13, c = 'r')
plt.plot(x, y23, c = 'b')
plt.xlabel('x2')
plt.ylabel('x1')

ax.scatter(1, 4, c='w', s=20)
ax.scatter(5, 1, c='w', s=20)
ax.scatter(0, 0, c='w', s=20)


plt.show()


