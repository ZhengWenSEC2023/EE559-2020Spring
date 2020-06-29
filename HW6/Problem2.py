# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:02:47 2020

@author: Lenovo
"""

import numpy as np

s1 = [[0, 0],
      [0, 1],
      [0, -1]
      ]

s2 = [[-2, 0],
      [-1, 0],
      [0,  2],
      [0, -2],
      [1, 0],
      [2, 0],
      ]

for each in s1:
    print(np.array([1, each[0], each[1], each[0] ** 2, each[0] * each[1], each[1] ** 2]))
    
print()
for each in s2:
    print(np.array([1, each[0], each[1], each[0] ** 2, each[0] * each[1], each[1] ** 2]))
    
print()
for each in s1:
    print(np.array([each[0] ** 2, each[1] ** 2]))
print()
for each in s2:
    print(np.array([each[0] ** 2, each[1] ** 2]))
