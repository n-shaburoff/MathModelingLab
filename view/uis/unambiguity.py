# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'unambiguity.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Unambiguity(object):
    def setupUi(self, Unambiguity):
        Unambiguity.setObjectName("Unambiguity")
        Unambiguity.resize(741, 306)
        self.centralwidget = QtWidgets.QWidget(Unambiguity)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 10, 681, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 80, 31, 21))
        self.label_2.setObjectName("label_2")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(50, 70, 161, 41))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(50, 150, 161, 41))
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 160, 31, 21))
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(250, 110, 106, 30))
        self.pushButton.setObjectName("pushButton")
        Unambiguity.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Unambiguity)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 741, 26))
        self.menubar.setObjectName("menubar")
        Unambiguity.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Unambiguity)
        self.statusbar.setObjectName("statusbar")
        Unambiguity.setStatusBar(self.statusbar)

        self.retranslateUi(Unambiguity)
        QtCore.QMetaObject.connectSlotsByName(Unambiguity)

    def retranslateUi(self, Unambiguity):
        _translate = QtCore.QCoreApplication.translate
        Unambiguity.setWindowTitle(_translate("Unambiguity", "MainWindow"))
        self.label.setText(_translate("Unambiguity", "Відповідь не однозначна, введіть вектори V0, VG для обрахунків"))
        self.label_2.setText(_translate("Unambiguity", "V0"))
        self.textEdit.setHtml(_translate("Unambiguity", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">3*t**2+2*sin(x)</p></body></html>"))
        self.textEdit_2.setHtml(_translate("Unambiguity", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">x**3+cos(t)</p></body></html>"))
        self.label_3.setText(_translate("Unambiguity", "VG"))
        self.pushButton.setText(_translate("Unambiguity", "Submit"))
