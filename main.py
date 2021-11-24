from model.main import Model
from view.main import start as viewStart

if __name__ == "__main__":
    model = Model()
    viewStart(model)