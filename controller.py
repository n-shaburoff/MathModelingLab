from MathModelingLab.graph import Graph
from MathModelingLab.math import Math
from graph import Plotter


def startCalculation(model):
    m = Math(model)
    m.initPandAv()
    # g = Graph(model, m)
    # g.build()
    g = Plotter(model.A, model.B, model.T)
    g.plot_3d(m.searchY, smp.lambdify(('x', 't'), model.Y), "Text")