from model.main import Model
from view.main import start as viewStart
from controller.math import startCalculation

if __name__ == "__main__":
    model = Model()
    viewStart(model)
