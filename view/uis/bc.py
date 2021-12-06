# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'boundary_conditions.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_boundaryConditions(object):
    def setupUi(self, boundaryConditions):
        boundaryConditions.setObjectName("boundaryConditions")
        boundaryConditions.resize(857, 586)
        self.centralwidget = QtWidgets.QWidget(boundaryConditions)
        self.centralwidget.setObjectName("centralwidget")
        self.labelTitle = QtWidgets.QLabel(self.centralwidget)
        self.labelTitle.setGeometry(QtCore.QRect(280, 20, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.labelTitle.setFont(font)
        self.labelTitle.setObjectName("labelTitle")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 90, 171, 31))
        self.label.setObjectName("label")
        self.condNumb = QtWidgets.QTextEdit(self.centralwidget)
        self.condNumb.setGeometry(QtCore.QRect(240, 80, 41, 41))
        self.condNumb.setObjectName("condNumb")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(340, 90, 301, 21))
        self.label_3.setObjectName("label_3")
        self.dotsNumb = QtWidgets.QTextEdit(self.centralwidget)
        self.dotsNumb.setGeometry(QtCore.QRect(640, 80, 41, 41))
        self.dotsNumb.setObjectName("dotsNumb")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(340, 150, 51, 31))
        self.pushButton.setObjectName("pushButton")
        boundaryConditions.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(boundaryConditions)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 857, 26))
        self.menubar.setObjectName("menubar")
        boundaryConditions.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(boundaryConditions)
        self.statusbar.setObjectName("statusbar")
        boundaryConditions.setStatusBar(self.statusbar)

        self.retranslateUi(boundaryConditions)
        QtCore.QMetaObject.connectSlotsByName(boundaryConditions)

    def retranslateUi(self, boundaryConditions):
        _translate = QtCore.QCoreApplication.translate
        boundaryConditions.setWindowTitle(_translate("boundaryConditions", "Крайові умови"))
        self.labelTitle.setText(_translate("boundaryConditions", "Крайові умови"))
        self.label.setText(_translate("boundaryConditions", "Кількість умов (R_r)"))
        self.label_3.setText(_translate("boundaryConditions", "Кількість точок дискретизації (L_г)"))
        self.pushButton.setText(_translate("boundaryConditions", "Ok"))
        self.condNumb.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>"))
        self.dotsNumb.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">6</p></body></html>"))


class AfterOk(object):
    def setupUi(self, boundaryConditions):
        boundaryConditions.getValues()
        boundaryConditions.setObjectName("boundaryConditions")
        boundaryConditions.resize(857, 586)
        self.centralwidget = QtWidgets.QWidget(boundaryConditions)
        self.centralwidget.setObjectName("centralwidget")
        self.labelTitle = QtWidgets.QLabel(self.centralwidget)
        self.labelTitle.setGeometry(QtCore.QRect(280, 20, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.labelTitle.setFont(font)
        self.labelTitle.setObjectName("labelTitle")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 90, 171, 31))
        self.label.setObjectName("label")
        self.condNumb = QtWidgets.QTextEdit(self.centralwidget)
        self.condNumb.setGeometry(QtCore.QRect(240, 80, 41, 41))
        self.condNumb.setObjectName("condNumb")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(340, 90, 301, 21))
        self.label_3.setObjectName("label_3")
        self.dotsNumb = QtWidgets.QTextEdit(self.centralwidget)
        self.dotsNumb.setGeometry(QtCore.QRect(640, 80, 41, 41))
        self.dotsNumb.setObjectName("dotsNumb")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(340, 150, 51, 31))
        self.pushButton.setObjectName("pushButton")
        boundaryConditions.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(boundaryConditions)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 857, 26))
        self.menubar.setObjectName("menubar")
        boundaryConditions.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(boundaryConditions)
        self.statusbar.setObjectName("statusbar")
        boundaryConditions.setStatusBar(self.statusbar)

        # creating label for l
        self.label_l = QtWidgets.QLabel(self.centralwidget)
        self.label_l.setGeometry(QtCore.QRect(20, 280, 171, 21))
        self.label_l.setObjectName("label_l")

        # fields for l
        self.l_inputs = []
        self.lx = 20
        self.ly = 300
        for i in range(int(boundaryConditions.conNumber)):
            self.l_inputs.append([])
            for j in range(1):
                self.l_inputs[i].append(QtWidgets.QTextEdit(self.centralwidget))
                self.l_inputs[i][j].setGeometry(QtCore.QRect(self.lx, self.ly, 120, 41))
                self.lx += 60

            self.ly += 60
            self.lx = 20

        # setting value for LG
        self.l_inputs[0][0].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>")


       # arrays for x and t lables
        self.x_labels = []
        self.t_labels = []
        self.y_labels = []

        # matrix for x values
        self.inputs = []
        self.inputs_x = 190
        self.inputs_y = 300

        # creating x labels
        self.xcount = 190
        for i in range(int(boundaryConditions.dotsNumber)):
            self.x_labels.append(QtWidgets.QLabel(self.centralwidget))
            self.x_labels[i].setGeometry(QtCore.QRect(self.xcount, 280, 171, 21))
            self.x_labels[i].setObjectName("label_x{}".format(i+1))
            self.x_labels[i].setText("x{}".format(i+1))
            self.xcount += 65

        # creating x fields
        for i in range(1):
            self.inputs.append([])
            for j in range(int(boundaryConditions.dotsNumber)):
                self.inputs[i].append(QtWidgets.QTextEdit(self.centralwidget))
                self.inputs[i][j].setGeometry(QtCore.QRect(self.inputs_x, self.inputs_y, 41, 41))
                self.inputs_x += 60

            self.inputs_y += 60
            self.inputs_x = 190

        # setting x values
        self.inputs[0][0].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">-1</p></body></html>")

        self.inputs[0][1].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">-1</p></body></html>")

        self.inputs[0][2].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">-1</p></body></html>")

        self.inputs[0][3].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>")

        self.inputs[0][4].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>")

        self.inputs[0][5].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>")

        self.tcount = self.xcount + 40
        # matrix for t values
        self.inputs_t = []
        self.tcval = self.tcount
        self.tx = self.tcval
        self.ty = 300

        # creating t labels
        for i in range(int(boundaryConditions.dotsNumber)):
            self.t_labels.append(QtWidgets.QLabel(self.centralwidget))
            self.t_labels[i].setGeometry(QtCore.QRect(self.tcount, 280, 171, 21))
            self.t_labels[i].setObjectName("label_t{}".format(i+1))
            self.t_labels[i].setText("t{}".format(i+1))
            self.tcount += 65

        # creating t fields
        for i in range(1):
            self.inputs_t.append([])
            for j in range(int(boundaryConditions.dotsNumber)):
                self.inputs_t[i].append(QtWidgets.QTextEdit(self.centralwidget))
                self.inputs_t[i][j].setGeometry(QtCore.QRect(self.tx, self.ty, 41, 41))
                self.tx += 60

            self.ty += 60
            self.tx = self.tcval

        # setting x values
        self.inputs_t[0][0].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>")

        self.inputs_t[0][1].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0.5</p></body></html>")

        self.inputs_t[0][2].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>")

        self.inputs_t[0][3].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>")

        self.inputs_t[0][4].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0.5</p></body></html>")

        self.inputs_t[0][5].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>")

        self.ycount = self.tcount + 40
        # matrix for y values
        self.inputs_y = []
        self.ycval = self.ycount
        self.yx = self.ycval
        self.yy = 300

        # creating y labels
        for i in range(int(boundaryConditions.dotsNumber)):
            self.y_labels.append(QtWidgets.QLabel(self.centralwidget))
            self.y_labels[i].setGeometry(QtCore.QRect(self.ycount, 280, 171, 21))
            self.y_labels[i].setObjectName("label_y{}".format(i+1))
            self.y_labels[i].setText("y{}".format(i+1))
            self.ycount += 65

        # creating y fields
        for i in range(int(boundaryConditions.conNumber)):
            self.inputs_y.append([])
            for j in range(int(boundaryConditions.dotsNumber)):
                self.inputs_y[i].append(QtWidgets.QTextEdit(self.centralwidget))
                self.inputs_y[i][j].setGeometry(QtCore.QRect(self.yx, self.yy, 41, 41))
                self.yx += 60

            self.yy += 60
            self.yx = self.ycval

        # setting y values
        self.inputs_y[0][0].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>")

        self.inputs_y[0][1].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0.25</p></body></html>")

        self.inputs_y[0][2].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>")

        self.inputs_y[0][3].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>")

        self.inputs_y[0][4].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0.25</p></body></html>")

        self.inputs_y[0][5].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>")


        # button for submitting form
        self.sb = QtWidgets.QPushButton(self.centralwidget)
        self.sb.setGeometry(QtCore.QRect(self.yx+30, self.yy+30, 70, 30))
        self.sb.setObjectName("submitButton")


        self.retranslateUi(boundaryConditions)
        QtCore.QMetaObject.connectSlotsByName(boundaryConditions)

    def retranslateUi(self, boundaryConditions):
        _translate = QtCore.QCoreApplication.translate
        boundaryConditions.setWindowTitle(_translate("boundaryConditions", "Крайові умови"))
        self.labelTitle.setText(_translate("boundaryConditions", "Крайові умови"))
        self.label.setText(_translate("boundaryConditions", "Кількість умов (R_r)"))
        self.label_3.setText(_translate("boundaryConditions", "Кількість точок дискретизації (L_г)"))
        self.label_l.setText(_translate("boundaryConditions", "L"))
        self.pushButton.setText(_translate("boundaryConditions", "Ok"))
        self.sb.setText(_translate("initial_conditions", "Submit"))
        self.condNumb.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>"))
        self.dotsNumb.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">6</p></body></html>"))

class MainModeBCUI(object):
    def setupUi(self, boundaryConditions):
        boundaryConditions.setObjectName("boundaryConditions")
        boundaryConditions.resize(857, 586)
        self.centralwidget = QtWidgets.QWidget(boundaryConditions)
        self.centralwidget.setObjectName("centralwidget")
        self.labelTitle = QtWidgets.QLabel(self.centralwidget)
        self.labelTitle.setGeometry(QtCore.QRect(280, 20, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.labelTitle.setFont(font)
        self.labelTitle.setObjectName("labelTitle")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 90, 171, 31))
        self.label.setObjectName("label")
        self.condNumb = QtWidgets.QTextEdit(self.centralwidget)
        self.condNumb.setGeometry(QtCore.QRect(240, 80, 41, 41))
        self.condNumb.setObjectName("condNumb")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(340, 90, 301, 21))
        self.label_3.setObjectName("label_3")
        self.dotsNumb = QtWidgets.QTextEdit(self.centralwidget)
        self.dotsNumb.setGeometry(QtCore.QRect(640, 80, 41, 41))
        self.dotsNumb.setObjectName("dotsNumb")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(340, 150, 51, 31))
        self.pushButton.setObjectName("pushButton")
        boundaryConditions.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(boundaryConditions)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 857, 26))
        self.menubar.setObjectName("menubar")
        boundaryConditions.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(boundaryConditions)
        self.statusbar.setObjectName("statusbar")
        boundaryConditions.setStatusBar(self.statusbar)

        self.retranslateUi(boundaryConditions)
        QtCore.QMetaObject.connectSlotsByName(boundaryConditions)

    def retranslateUi(self, boundaryConditions):
        _translate = QtCore.QCoreApplication.translate
        boundaryConditions.setWindowTitle(_translate("boundaryConditions", "Крайові умови"))
        self.labelTitle.setText(_translate("boundaryConditions", "Крайові умови"))
        self.label.setText(_translate("boundaryConditions", "Кількість умов (R_r)"))
        self.label_3.setText(_translate("boundaryConditions", "Кількість точок дискретизації (L_г)"))
        self.pushButton.setText(_translate("boundaryConditions", "Ok"))

class MainModeAfterOKBCUI(object):
    def setupUi(self, boundaryConditions):
        boundaryConditions.getValues()
        boundaryConditions.setObjectName("boundaryConditions")
        boundaryConditions.resize(857, 586)
        self.centralwidget = QtWidgets.QWidget(boundaryConditions)
        self.centralwidget.setObjectName("centralwidget")
        self.labelTitle = QtWidgets.QLabel(self.centralwidget)
        self.labelTitle.setGeometry(QtCore.QRect(280, 20, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.labelTitle.setFont(font)
        self.labelTitle.setObjectName("labelTitle")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 90, 171, 31))
        self.label.setObjectName("label")
        self.condNumb = QtWidgets.QTextEdit(self.centralwidget)
        self.condNumb.setGeometry(QtCore.QRect(240, 80, 41, 41))
        self.condNumb.setObjectName("condNumb")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(340, 90, 301, 21))
        self.label_3.setObjectName("label_3")
        self.dotsNumb = QtWidgets.QTextEdit(self.centralwidget)
        self.dotsNumb.setGeometry(QtCore.QRect(640, 80, 41, 41))
        self.dotsNumb.setObjectName("dotsNumb")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(340, 150, 51, 31))
        self.pushButton.setObjectName("pushButton")
        boundaryConditions.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(boundaryConditions)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 857, 26))
        self.menubar.setObjectName("menubar")
        boundaryConditions.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(boundaryConditions)
        self.statusbar.setObjectName("statusbar")
        boundaryConditions.setStatusBar(self.statusbar)

        # creating label for l
        self.label_l = QtWidgets.QLabel(self.centralwidget)
        self.label_l.setGeometry(QtCore.QRect(20, 280, 171, 21))
        self.label_l.setObjectName("label_l")

        # fields for l
        self.l_inputs = []
        self.lx = 20
        self.ly = 300
        for i in range(int(boundaryConditions.conNumber)):
            self.l_inputs.append([])
            for j in range(1):
                self.l_inputs[i].append(QtWidgets.QTextEdit(self.centralwidget))
                self.l_inputs[i][j].setGeometry(QtCore.QRect(self.lx, self.ly, 120, 41))
                self.lx += 60

            self.ly += 60
            self.lx = 20

       # arrays for x and t lables
        self.x_labels = []
        self.t_labels = []
        self.y_labels = []

        # matrix for x values
        self.inputs = []
        self.inputs_x = 190
        self.inputs_y = 300

        # creating x labels
        self.xcount = 190
        for i in range(int(boundaryConditions.dotsNumber)):
            self.x_labels.append(QtWidgets.QLabel(self.centralwidget))
            self.x_labels[i].setGeometry(QtCore.QRect(self.xcount, 280, 171, 21))
            self.x_labels[i].setObjectName("label_x{}".format(i+1))
            self.x_labels[i].setText("x{}".format(i+1))
            self.xcount += 65

        # creating x fields
        for i in range(1):
            self.inputs.append([])
            for j in range(int(boundaryConditions.dotsNumber)):
                self.inputs[i].append(QtWidgets.QTextEdit(self.centralwidget))
                self.inputs[i][j].setGeometry(QtCore.QRect(self.inputs_x, self.inputs_y, 41, 41))
                self.inputs_x += 60

            self.inputs_y += 60
            self.inputs_x = 190

        self.tcount = self.xcount + 40
        # matrix for t values
        self.inputs_t = []
        self.tcval = self.tcount
        self.tx = self.tcval
        self.ty = 300

        # creating t labels
        for i in range(int(boundaryConditions.dotsNumber)):
            self.t_labels.append(QtWidgets.QLabel(self.centralwidget))
            self.t_labels[i].setGeometry(QtCore.QRect(self.tcount, 280, 171, 21))
            self.t_labels[i].setObjectName("label_t{}".format(i+1))
            self.t_labels[i].setText("t{}".format(i+1))
            self.tcount += 65

        # creating t fields
        for i in range(1):
            self.inputs_t.append([])
            for j in range(int(boundaryConditions.dotsNumber)):
                self.inputs_t[i].append(QtWidgets.QTextEdit(self.centralwidget))
                self.inputs_t[i][j].setGeometry(QtCore.QRect(self.tx, self.ty, 41, 41))
                self.tx += 60

            self.ty += 60
            self.tx = self.tcval

        self.ycount = self.tcount + 40
        # matrix for y values
        self.inputs_y = []
        self.ycval = self.ycount
        self.yx = self.ycval
        self.yy = 300

        # creating y labels
        for i in range(int(boundaryConditions.dotsNumber)):
            self.y_labels.append(QtWidgets.QLabel(self.centralwidget))
            self.y_labels[i].setGeometry(QtCore.QRect(self.ycount, 280, 171, 21))
            self.y_labels[i].setObjectName("label_y{}".format(i+1))
            self.y_labels[i].setText("y{}".format(i+1))
            self.ycount += 65

        # creating y fields
        for i in range(int(boundaryConditions.conNumber)):
            self.inputs_y.append([])
            for j in range(int(boundaryConditions.dotsNumber)):
                self.inputs_y[i].append(QtWidgets.QTextEdit(self.centralwidget))
                self.inputs_y[i][j].setGeometry(QtCore.QRect(self.yx, self.yy, 41, 41))
                self.yx += 60

            self.yy += 60
            self.yx = self.ycval

        # button for submitting form
        self.sb = QtWidgets.QPushButton(self.centralwidget)
        self.sb.setGeometry(QtCore.QRect(self.yx+30, self.yy+30, 70, 30))
        self.sb.setObjectName("submitButton")


        self.retranslateUi(boundaryConditions)
        QtCore.QMetaObject.connectSlotsByName(boundaryConditions)

    def retranslateUi(self, boundaryConditions):
        _translate = QtCore.QCoreApplication.translate
        boundaryConditions.setWindowTitle(_translate("boundaryConditions", "Крайові умови"))
        self.labelTitle.setText(_translate("boundaryConditions", "Крайові умови"))
        self.label.setText(_translate("boundaryConditions", "Кількість умов (R_r)"))
        self.label_3.setText(_translate("boundaryConditions", "Кількість точок дискретизації (L_г)"))
        self.label_l.setText(_translate("boundaryConditions", "L"))
        self.pushButton.setText(_translate("boundaryConditions", "Ok"))
        self.sb.setText(_translate("initial_conditions", "Submit"))

