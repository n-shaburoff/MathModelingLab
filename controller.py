from MathModelingLab.graph import Graph
from MathModelingLab.math import Math


def startCalculation(model):
    m = Math(model)
    m.initPandAv()
    g = Graph(model, m)
    g.build()