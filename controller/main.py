from math import Math
from graph import Graph


def startCalculation(model):
    m = Math(model)
    g = Graph(model, m)
    g.build()
