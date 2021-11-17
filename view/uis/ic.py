# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'initial_conditions.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

global conNumber, dotsNumber, derNumber


class Ui_initial_conditions(object):
    def setupUi(self, initial_conditions):
        initial_conditions.setObjectName("initial_conditions")
        initial_conditions.resize(917, 508)
        self.centralwidget = QtWidgets.QWidget(initial_conditions)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(260, 20, 421, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 100, 171, 21))
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(370, 140, 61, 30))
        self.pushButton.setObjectName("pushButton")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 150, 301, 21))
        self.label_4.setObjectName("label_4")
        self.condNum = QtWidgets.QTextEdit(self.centralwidget)
        self.condNum.setGeometry(QtCore.QRect(200, 90, 41, 41))
        self.condNum.setObjectName("condNum")
        self.dotsNum = QtWidgets.QTextEdit(self.centralwidget)
        self.dotsNum.setGeometry(QtCore.QRect(320, 140, 41, 41))
        self.dotsNum.setObjectName("dotsNum")
        initial_conditions.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(initial_conditions)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 917, 26))
        self.menubar.setObjectName("menubar")
        initial_conditions.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(initial_conditions)
        self.statusbar.setObjectName("statusbar")
        initial_conditions.setStatusBar(self.statusbar)

        self.retranslateUi(initial_conditions)
        QtCore.QMetaObject.connectSlotsByName(initial_conditions)

    def retranslateUi(self, initial_conditions):
        _translate = QtCore.QCoreApplication.translate
        initial_conditions.setWindowTitle(_translate("initial_conditions", "Початково-крайовий стан"))
        self.label.setText(_translate("initial_conditions", "Початково-крайові умови"))
        self.label_3.setText(_translate("initial_conditions", "Кількість умов (R_0)"))
        self.pushButton.setText(_translate("initial_conditions", "Ok"))
        self.label_4.setText(_translate("initial_conditions", "Кількість точок дискретизації (L_0)"))
        self.condNum.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2</p></body></html>"))
        self.dotsNum.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2</p></body></html>"))


class Ui_AfterButton_Clicked(object):
    def setupUi(self, initial_conditions):
        initial_conditions.getValues()
        initial_conditions.setObjectName("initial_conditions")
        initial_conditions.resize(917, 508)
        self.centralwidget = QtWidgets.QWidget(initial_conditions)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(260, 20, 421, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 100, 171, 21))
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(370, 140, 61, 30))
        self.pushButton.setObjectName("pushButton")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 150, 301, 21))
        self.label_4.setObjectName("label_4")
        self.condNum = QtWidgets.QTextEdit(self.centralwidget)
        self.condNum.setGeometry(QtCore.QRect(200, 90, 41, 41))
        self.condNum.setObjectName("condNum")
        self.dotsNum = QtWidgets.QTextEdit(self.centralwidget)
        self.dotsNum.setGeometry(QtCore.QRect(320, 140, 41, 41))
        self.dotsNum.setObjectName("dotsNum")
        initial_conditions.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(initial_conditions)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 917, 26))
        self.menubar.setObjectName("menubar")
        initial_conditions.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(initial_conditions)
        self.statusbar.setObjectName("statusbar")
        initial_conditions.setStatusBar(self.statusbar)

        # creating label for l
        self.label_l = QtWidgets.QLabel(self.centralwidget)
        self.label_l.setGeometry(QtCore.QRect(20, 280, 171, 21))
        self.label_l.setObjectName("label_l")

        # creating fields for l
        self.l_inputs = []
        self.lx = 20
        self.ly = 300
        for i in range(int(initial_conditions.conNumber)):
            self.l_inputs.append([])
            for j in range(1):
                self.l_inputs[i].append(QtWidgets.QTextEdit(self.centralwidget))
                self.l_inputs[i][j].setGeometry(QtCore.QRect(self.lx, self.ly, 120, 41))
                self.lx += 60

            self.ly += 60
            self.lx = 20

        # setting value for LO
        self.l_inputs[0][0].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2dx</p></body></html>")

        self.l_inputs[1][0].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">4dx+3dx**2</p></body></html>")

        # arrays for x and y lables
        self.x_labels = []
        self.y_labels = []

        # matrix for x values
        self.inputs = []
        self.inputs_x = 190
        self.inputs_y = 300

        # creating x labels
        self.xcount = 190
        for i in range(int(initial_conditions.dotsNumber)):
            self.x_labels.append(QtWidgets.QLabel(self.centralwidget))
            self.x_labels[i].setGeometry(QtCore.QRect(self.xcount, 280, 171, 21))
            self.x_labels[i].setObjectName("label_x{}".format(i+1))
            self.x_labels[i].setText("x{}".format(i+1))
            self.xcount += 65

        # creating x fields
        for i in range(1):
            self.inputs.append([])
            for j in range(int(initial_conditions.dotsNumber)):
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
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2</p></body></html>")

        self.inputs[0][1].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">3</p></body></html>")


        self.ycount = self.xcount + 40
        # matrix for y values
        self.inputs_y = []
        self.ycval = self.ycount
        self.yx = self.ycval
        self.yy = 300

        # creating y labels
        for i in range(int(initial_conditions.dotsNumber)):
            self.y_labels.append(QtWidgets.QLabel(self.centralwidget))
            self.y_labels[i].setGeometry(QtCore.QRect(self.ycount, 280, 171, 21))
            self.y_labels[i].setObjectName("label_y{}".format(i+1))
            self.y_labels[i].setText("y{}".format(i+1))
            self.ycount += 65

        # creating y fields
        for i in range(int(initial_conditions.conNumber)):
            self.inputs_y.append([])
            for j in range(int(initial_conditions.dotsNumber)):
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
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">3</p></body></html>")

        self.inputs_y[0][1].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">12</p></body></html>")

        self.inputs_y[1][0].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">43</p></body></html>")

        self.inputs_y[1][1].setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">53</p></body></html>")

        # button for submitting form
        self.sb = QtWidgets.QPushButton(self.centralwidget)
        self.sb.setGeometry(QtCore.QRect(self.yx+30, self.yy+30, 70, 30))
        self.sb.setObjectName("submitButton")

        self.retranslateUi(initial_conditions)
        QtCore.QMetaObject.connectSlotsByName(initial_conditions)

    def retranslateUi(self, initial_conditions):
        _translate = QtCore.QCoreApplication.translate
        initial_conditions.setWindowTitle(_translate("initial_conditions", "Початково-крайовий стан"))
        self.label.setText(_translate("initial_conditions", "Початкові умови"))
        self.label_3.setText(_translate("initial_conditions", "Кількість умов (R_0)"))
        self.pushButton.setText(_translate("initial_conditions", "Ok"))
        self.label_4.setText(_translate("initial_conditions", "Кількість точок дискретизації (L_0)"))
        self.label_l.setText(_translate("initial_conditions", "L"))
        self.sb.setText(_translate("initial_conditions", "Submit"))
        self.condNum.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2</p></body></html>"))
        self.dotsNum.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2</p></body></html>"))

