from typing import Text
from PyQt5 import QtWidgets, QtCore, QtGui

from uis.mode import Ui_MainWindow as Mode
from uis.mat_mode import Ui_MainWindow as MM
from uis.ic import Ui_initial_conditions as IC
from uis.ic import Ui_AfterButton_Clicked as AIC
from uis.bc import AfterOk, Ui_boundaryConditions as BC
from uis.unambiguity import Ui_Unambiguity as UNAM
from uis.mat_mode import MainModeMathModelingUI as MMM
from uis.ic import MainModeInitialConditionsUI as MIC
from uis.ic import MainModeAfterOkUI as MAIC
from uis.bc import MainModeBCUI as MBC
from uis.bc import MainModeAfterOKBCUI as MABC
from uis.unambiguity import MainModeUnambiguityUI as MUNAM
from model import Model

import sys


class ChooseMethod(QtWidgets.QMainWindow):
    def __init__(self, model):
        super(ChooseMethod, self).__init__()
        self.ui = Mode()
        self.ui.setupUi(self)

        self.tm = TestMatModeling(model)
        self.mm = MainModeMathModeling(model)

        self.ui.radioButton.toggled.connect(self.OpenModeWindow)
        self.ui.radioButton_2.toggled.connect(self.OpenModeWindow)

    def OpenModeWindow(self):
        radioButton = self.sender()
        if radioButton.isChecked() and radioButton.text() == "Тестовий":
            self.tm.show()
            self.close()
        else:
            self.mm.show()
            self.close()

class MainModeMathModeling(QtWidgets.QMainWindow):
    def __init__(self, model = Model()):
        super(MainModeMathModeling, self).__init__()

        #ui
        self.ui = MMM()
        self.ui.setupUi(self)

        #initial conditions
        self.ic = MainModeInitialConditions(model)

        # math model
        self.model = model


        self.ui.pushButton.clicked.connect(self.saveValuesAndOpenMainIC)

    def saveValuesAndOpenMainIC(self):
        self.model.L = str(self.ui.comboBox.currentText())
        self.model.U = str(self.ui.uText.toPlainText())
        self.model.A = str(self.ui.AText.toPlainText())
        self.model.B = str(self.ui.BText.toPlainText())
        self.model.T = str(self.ui.TText.toPlainText())
        self.model.Y = str(self.ui.yText.toPlainText())

        print(self.model)

        self.ic.show()
        self.close()


class TestMatModeling(QtWidgets.QMainWindow):
    def __init__(self, model = Model()):
        super(TestMatModeling, self).__init__()
        # class ui
        self.ui = MM()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.saveAndOpenIC)

        # initial conditions
        self.ic = InitialConditions(model)

        # mathematical model
        self.model = model

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

class MainModeInitialConditions(QtWidgets.QMainWindow):
    def __init__(self, model = Model()):
        super(MainModeInitialConditions, self).__init__()
        self.ui = MIC()
        self.aui = MAIC()

        self.bc = MainModeBC(model)

        self.conNumber = ''
        self.dotsNumber = ''

        self.model = model

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


class InitialConditions(QtWidgets.QMainWindow):
    def __init__(self, model = Model()):
        super(InitialConditions, self).__init__()
        self.ui = IC()
        self.aui = AIC()
        self.bc = BoundaryConditions(model)

        self.conNumber = ''
        self.dotsNumber = ''

        self.model = model

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

class MainModeBC(QtWidgets.QMainWindow):
    def __init__(self, model = Model()):
        super(MainModeBC, self).__init__()
        self.ui = MBC()
        self.aui = MABC()

        self.conNumber = ''
        self.dotsNumber = ''

        self.unam = MainModeUnambiguity(model)

        self.model = model

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

class BoundaryConditions(QtWidgets.QMainWindow):
    def __init__(self, model = Model()):
        super(BoundaryConditions, self).__init__()
        self.ui = BC()
        self.aui = AfterOk()

        self.conNumber = ''
        self.dotsNumber = ''

        self.unam = Unambiguity(model)

        self.model = model

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

class MainModeUnambiguity(QtWidgets.QMainWindow):
    def __init__(self, model = Model()):
        super(MainModeUnambiguity, self).__init__()
        self.ui = MUNAM()
        self.ui.setupUi(self)

        # math model
        self.model = model

        self.ui.pushButton.clicked.connect(self.getV0andVG)

    def getV0andVG(self):
        self.model.V0 = str(self.ui.V0.toPlainText())
        print(self.model.V0)
        self.model.VG = str(self.ui.VG.toPlainText())
        print(self.model.VG)

class Unambiguity(QtWidgets.QMainWindow):
    def __init__(self, model = Model()):
        super(Unambiguity, self).__init__()
        self.ui = UNAM()
        self.ui.setupUi(self)

        # math model
        self.model = model

        self.ui.pushButton.clicked.connect(self.getV0andVG)

    def getV0andVG(self):
        self.model.V0 = str(self.ui.V0.toPlainText())
        print(self.model.V0)
        self.model.VG = str(self.ui.VG.toPlainText())
        print(self.model.VG)
        self.close()


def start(model):
    app = QtWidgets.QApplication(sys.argv)
    w = ChooseMethod(model)
    w.show()
    app.exec_()