# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:03:50 2020

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt

m1 = np.array([1, 2])
m2 = np.array([1, -1])
m3 = np.array([-2, 2])

sig = np.array([[1, -1], [-1, 2]])

step = 0.005
x = np.arange(-4, 4 + step, step)
y = np.arange(-4, 4 + step, step)
xrange = (x[0], x[-1])
yrange = (y[0], y[-1])

X1, X2 = np.meshgrid(x,y)

Z1 = 2*(X1 - m1[0])**2 + 2*(X1 - m1[0])*(X2 - m1[1]) + (X2 - m1[1])**2 - 1
Z2 = 2*(X1 - m2[0])**2 + 2*(X1 - m2[0])*(X2 - m2[1]) + (X2 - m2[1])**2 - 1
Z3 = 2*(X1 - m3[0])**2 + 2*(X1 - m3[0])*(X2 - m3[1]) + (X2 - m3[1])**2 - 1

g1 = 4*X1 + 3*X2 - 5
g2 = X1 - 1/2
g3 = -2 * X1 - 2

dec1 = (g1 > g2) & (g1 > g3)
dec2 = (g2 >= g1) & (g2 > g3)
dec3 = (g3 >= g1) & (g3 >= g2)

dec_map = np.zeros(dec1.shape)
for i in range(dec_map.shape[0]):
    for j in range(dec_map.shape[1]):
        if dec1[i, j]:
            dec_map[i, j] = 1
        if dec2[i, j]:
            dec_map[i, j] = 2
        if dec3[i, j]:
            dec_map[i, j] = 3

dec_map = np.flipud(dec_map)
plt.contour(X1, X2, Z1, 0, colors='r')
plt.contour(X1, X2, Z2, 0, colors='g')
plt.contour(X1, X2, Z3, 0, colors='b')
plt.plot(m1[0], m1[1], 'ro', label='m1')
plt.plot(m2[0], m2[1], 'go', label='m2')
plt.plot(m3[0], m3[1], 'bo', label='m3')
plt.imshow(dec_map, extent=[xrange[0], xrange[1], yrange[0], yrange[1]])
plt.plot(x, 3/2 - x, color='c', lw=3, label='g12')
plt.vlines(-1/2, -4, 4, colors='m', lw=3, label='g23')
plt.plot(x, -2 * x + 1, color='w', lw=3, label='g13')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.legend()
