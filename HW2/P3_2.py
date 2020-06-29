# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:03:05 2020

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt


def coeffCal(u1, u2):
    slot = [(-2)*u1[0]+2*u2[0], (-2)*u1[1]+2*u2[1]]
    inter = [u1[0]**2-u2[0]**2, u1[1]**2-u2[1]**2]
    return slot, inter

u11 = [2, 0]
u12 = [0, -2]
slot1, inter1 = coeffCal(u11, u12)


x = np.linspace(-3, 4, 1000)
y23 = 2*x-(3/2)
y12 = np.array([-0.5]*1000)
y13 = -x
y_T = np.array([7]*1000)
y_B = np.array([-8]*1000)

ax = plt.gca()

gamma1 = plt.fill_between(x, y_B, np.append(y12[x<0.5], y13[x>=0.5]), facecolor = "green")
gamma2 = plt.fill_between(x, y_T, np.append(y12[x<0.5], y23[x>=0.5]), facecolor = "blue")
gamma3 = plt.fill_between(x[x>0.5], y23[x>0.5], y13[x>0.5], facecolor = "red")

plt.legend(handles=[gamma1, gamma2, gamma3], labels=["gamma1", "gamma2", "gamma3"])
plt.plot(x, y12, c = "g")
plt.plot(x, y13, c = 'r')
plt.plot(x, y23, c = 'b')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
