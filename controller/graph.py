import matplotlib.pyplot as plt
import numpy as np
import sympy as smp


def fun(x, y):
    return (x ** 2 - y ** 2) - x * y


fig = plt.figure()
smpFunc = smp.lambdify(('x', 'y'), '0.26*(x**2 - y**2)-x*y')

# #plot first
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-20, 20, 1)
X, Y = np.meshgrid(x, y)

zs = np.array([fun(x, y) for x, y in zip(X, Y)])
zs2 = np.array([smpFunc(x, y) for x, y in zip(X, Y)])

Z = zs.reshape(X.shape)
Z2 = zs2.reshape(X.shape)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# plot second
ax.plot_surface(X, Y, Z2)
ax.plot_surface(X, Y, Z)

plt.show()


