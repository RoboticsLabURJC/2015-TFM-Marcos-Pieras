# -*- coding: utf-8 -*-

from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy
import sys
import os.path as osp

path = osp.join(osp.dirname(sys.modules[__name__].__file__), 'logoJdeRobot.png')

class Gui(QtGui.QWidget):

    IMAGE_COLS_MAX=640
    IMAGE_ROWS_MAX=360
    updGUI=QtCore.pyqtSignal()

    def __init__(self, parent=None):

        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle("CamaraReal")
        self.setWindowIcon(QtGui.QIcon(path))
        self.resize(500,500)
        self.updGUI.connect(self.update)
        

        # Creo BoxLayout
        hbox = QtGui.QVBoxLayout()
     
        TestButton=QtGui.QPushButton("Evaluar")
        TestButton.clicked.connect(self.effect)

        self.imgLabel = QtGui.QLabel(self)
        hbox.addWidget(self.imgLabel)
        hbox.addWidget(TestButton)
        self.setLayout(hbox) 



    def setControl(self,control):
        self.control=control

    def update(self):
        #print 'updgui'
        image = self.control.getImage()
        if image != None:
            img = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
            size=QtCore.QSize(image.shape[1],image.shape[0])
            #self.imgLabel.resize(size)
            #self.resize(size)
            self.imgLabel.setPixmap(QtGui.QPixmap.fromImage(img))
            #print 'printimg'

    def effect(self):
        print 'shooooooooooooooooooooooooooooooooo'
        self.control.effect()

