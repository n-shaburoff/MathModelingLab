import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from MathModelingLab.math import Math
from MathModelingLab.model import Model


class Graph():

    def __init__(self, model: Model, m: Math):
        self.fig = plt.figure()
        self.math = m
        self.smpFunc = smp.lambdify(('x', 't'), model.Y)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.x = np.linspace(float(model.A), float(model.B), 6)
        self.y = np.linspace(0, float(model.T), 6)
        self.X = np.array(np.meshgrid(self.x, self.y))
        self.Y = np.array(np.meshgrid(self.x, self.y))


    def build(self):
        print("-------")
        print(self.x)
        print(self.y)
        zs = np.array([[self.math.searchY(xi, yi) for xi in self.x] for yi in self.y])
        zs2 = np.array([[self.smpFunc(xi, yi) for xi in self.x] for yi in self.y])

        Z = zs.reshape(self.X.shape)
        Z2 = zs2.reshape(self.X.shape)

        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')

        # plot second
        self.ax.plot_surface(self.X, self.Y, Z2)
        self.ax.plot_surface(self.X, self.Y, Z)

        plt.show()


    def buildGraph(self):
        fig = plt.figure()

        # #plot first
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-20, 20, 1)
        X, Y = np.meshgrid(x, y)

        zs = np.array([self.math.searchY(x, y) for x, y in zip(X, Y)])
        zs2 = np.array([self.smpFunc(x, y) for x, y in zip(X, Y)])

        Z = zs.reshape(X.shape)
        Z2 = zs2.reshape(X.shape)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # plot second
        ax.plot_surface(X, Y, Z2)
        ax.plot_surface(X, Y, Z)

        plt.show()