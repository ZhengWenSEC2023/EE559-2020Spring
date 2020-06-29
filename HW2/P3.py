# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:02:33 2020

@author: Lenovo
"""


import numpy as np
from plotDecBoundaries import plotDecBoundariesMVM

u1 = np.array([[0, -2], [0, 1]])

plotDecBoundariesMVM(u1)

u2 = np.array([[0, -2], [0, 1], [2, 0]])

plotDecBoundariesMVM(u2)
