import my_math as MATH
import graph
import sympy as smp


def startCalculation(model):
    m = MATH.Math(model)
    m.initPandAv()
    # g = Graph(model, m)
    # g.build()
    g = graph.Plotter(model.A, model.B, model.T)
    g.plot_3d(m.searchY, smp.lambdify(('x', 't'), model.Y), "Text")