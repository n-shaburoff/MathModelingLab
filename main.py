from model.main import Model
from view.main import start as viewStart
import controller

if __name__ == "__main__":
    model = Model()
    viewStart(model)
    controller.startCalculation(model)
