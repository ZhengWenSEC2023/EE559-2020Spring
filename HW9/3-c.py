import numpy as np
import matplotlib.pyplot as plt

m1 = np.array([1, 4])
m2 = np.array([4, 2])

ps1 = 0.8
ps2 = 0.2

l1 = 1
l2 = 10

step = 0.005
x = np.arange(0, 6 + step, step)
y = np.arange(0, 6 + step, step)
xrange = (x[0], x[-1])
yrange = (y[0], y[-1])

X1, X2 = np.meshgrid(x,y)

Z1 = (2/3) * (4*(X1 - m1[0])**2 + 2*(X1 - m1[0])*(X2 - m1[1]) + (X2 - m1[1])**2 - 1)
Z2 = (2/3) * (4*(X1 - m2[0])**2 + 2*(X1 - m2[0])*(X2 - m2[1]) + (X2 - m2[1])**2 - 1)

g1 = np.log(l1) + np.log(ps1) - (1/3) * (-16 * X1 - 10 * X2 + 28)
g2 = np.log(l2) + np.log(ps2) - (1/3) * (-36 * X1 - 12 * X2 + 84)

dec1 = g1 > g2
dec2 = g1 < g2

dec_map = np.zeros(dec1.shape)
for i in range(dec_map.shape[0]):
    for j in range(dec_map.shape[1]):
        if dec1[i, j]:
            dec_map[i, j] = 1
        if dec2[i, j]:
            dec_map[i, j] = 2

dec_map = np.flipud(dec_map)

plt.contour(X1, X2, Z1, 0, colors='r')
plt.contour(X1, X2, Z2, 0, colors='g')
plt.plot(m1[0], m1[1], 'ro', label='m1')
plt.plot(m2[0], m2[1], 'go', label='m2')
plt.imshow(dec_map, extent=[xrange[0], xrange[1], yrange[0], yrange[1]])
plt.plot(x, -10*x+28+3/2*np.log(0.4), color='w', label='g12', lw=3)
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.legend()
