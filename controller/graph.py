import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from math import Math
from MathModelingLab.main import Model


class Graph():

    def __init__(self, model: Model, m: Math):
        self.fig = plt.figure()
        self.math = m
        self.smpFunc = smp.lambdify(('x', 'y'), model.Y)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.x = np.linspace(model.A, model.B, 5)
        self.y = np.linspace(0, model.T, 5)
        self.X = np.meshgrid(self.x, self.y)
        self.Y = np.meshgrid(self.x, self.y)


    def build(self):
        zs = np.array([self.math.searchYG(x, y) for x, y in zip(self.X, self.Y)])
        zs2 = np.array([self.smpFunc(x, y) for x, y in zip(self.X, self.Y)])

        Z = zs.reshape(self.X.shape)
        Z2 = zs2.reshape(self.X.shape)

        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')

        # plot second
        self.ax.plot_surface(self.X, self.Y, Z2)
        self.ax.plot_surface(self.X, self.Y, Z)

        plt.show()