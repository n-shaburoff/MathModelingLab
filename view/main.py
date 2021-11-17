from PyQt5 import QtWidgets, QtCore, QtGui

from uis.mode import Ui_MainWindow as Mode
from uis.mat_mode import Ui_MainWindow as MM
from uis.ic import Ui_initial_conditions as IC
from uis.ic import Ui_AfterButton_Clicked as AIC
from uis.bc import AfterOk, Ui_boundaryConditions as BC
from uis.unambiguity import Ui_Unambiguity as UNAM

import sys


class ChooseMethod(QtWidgets.QMainWindow):
    def __init__(self):
        super(ChooseMethod, self).__init__()
        self.ui = Mode()
        self.ui.setupUi(self)

        self.ui.radioButton.toggled.connect(self.OpenTestModeWindow)

        self.tm = TestMatModeling()

    def OpenTestModeWindow(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.tm.show()
            self.close()

class TestMatModeling(QtWidgets.QMainWindow):
    def __init__(self):
        super(TestMatModeling, self).__init__()
        # class ui
        self.ui = MM()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.saveAndOpenIC)

        # initial conditions
        self.ic = InitialConditions()

        # mathematical model
        self.model = Model()

    def saveAndOpenIC(self):
        self.model.L = str(self.ui.comboBox.currentText())
        self.model.U = str(self.ui.uText.toPlainText())
        self.model.A = str(self.ui.AText.toPlainText())
        self.model.B = str(self.ui.BText.toPlainText())
        self.model.T = str(self.ui.TText.toPlainText())
        self.model.Y = str(self.ui.yText.toPlainText())

        print(self.model)

        self.ic.show()
        self.close()

class InitialConditions(QtWidgets.QMainWindow):
    def __init__(self):
        super(InitialConditions, self).__init__()
        self.ui = IC()
        self.aui = AIC()
        self.bc = BoundaryConditions()

        self.conNumber = ''
        self.dotsNumber = ''

        self.model = Model()

        self.startInitialUI()


    def saveAndOpenBC(self):
        # getting L0
        for i in range(int(self.conNumber)):
            for j in range(1):
                self.model.L0.append(str(self.aui.l_inputs[i][j].toPlainText()))

        # getting X0
        for i in range(1):
            for j in range(int(self.dotsNumber)):
                self.model.X0.append(str(self.aui.inputs[i][j].toPlainText()))

        # getting Y0
        for i in range(int(self.conNumber)):
            self.model.Y0.append([])
            for j in range(int(self.dotsNumber)):
                self.model.Y0[i].append(str(self.aui.inputs_y[i][j].toPlainText()))

        print(self.model)

        self.bc.show()
        self.close()

    def getValues(self):
        self.conNumber = str(self.ui.condNum.toPlainText())
        self.dotsNumber = str(self.ui.dotsNum.toPlainText())

    def startInitialUI(self):
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.startAfterUI)

    def startAfterUI(self):
        self.aui.setupUi(self)
        self.aui.sb.clicked.connect(self.saveAndOpenBC)

class BoundaryConditions(QtWidgets.QMainWindow):
    def __init__(self):
        super(BoundaryConditions, self).__init__()
        self.ui = BC()
        self.aui = AfterOk()

        self.conNumber = ''
        self.dotsNumber = ''

        self.unam = Unambiguity()

        self.model = Model()

        self.startBeforeOkUI()

    def saveValues(self):
        # getting LG
        for i in range(int(self.conNumber)):
            for j in range(1):
                self.model.LG.append(str(self.aui.l_inputs[i][j].toPlainText()))

        # getting XG
        for i in range(1):
            for j in range(int(self.dotsNumber)):
                self.model.XG.append(str(self.aui.inputs[i][j].toPlainText()))

        # getting TG
        for i in range(1):
            for j in range(int(self.dotsNumber)):
                self.model.TG.append(str(self.aui.inputs_t[i][j].toPlainText()))

        # getting YG
        for i in range(int(self.conNumber)):
            self.model.YG.append([])
            for j in range(int(self.dotsNumber)):
                self.model.YG[i].append(str(self.aui.inputs_y[i][j].toPlainText()))

        print(self.model)
        self.unam.show()
        self.close()

    def startBeforeOkUI(self):
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.startAfterOkUI)

    def startAfterOkUI(self):
        self.aui.setupUi(self)
        self.aui.sb.clicked.connect(self.saveValues)

    def getValues(self):
        self.conNumber = str(self.ui.condNumb.toPlainText())
        self.dotsNumber = str(self.ui.dotsNumb.toPlainText())

class Unambiguity(QtWidgets.QMainWindow):
    def __init__(self):
        super(Unambiguity, self).__init__()
        self.ui = UNAM()
        self.ui.setupUi(self)

class Model():
    def __init__(self):
        self.A = ""
        self.B = ""
        self.U = ""
        self.Y = ""
        self.G = ""
        self.L = ""
        self.T = ""
        self.L0 = []
        self.LG = []
        self.X0 = []
        self.XG = []
        self.Y0 = []
        self.YG = []
        self.TG = []

    def __str__(self):
        return f'A: {self.A},B :{self.B},U: {self.U},y^ = {self.Y},G: {self.G},L: {self.L},T: {self.T},L0: {self.L0}, X0: {self.X0}, Y0: {self.Y0},LG: {self.LG}, XG: {self.XG},TG: {self.TG}, YG: {self.YG}'

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = ChooseMethod()
    w.show()
    sys.exit(app.exec_())