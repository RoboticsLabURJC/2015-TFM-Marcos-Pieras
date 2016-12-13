# -*- coding: utf-8 -*-

from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy as np
import sys
import os.path as osp
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
import matplotlib.pyplot as plt


path = osp.join(osp.dirname(sys.modules[__name__].__file__), 'logoJdeRobot.png')

class Gui(QtGui.QWidget):

    IMAGE_COLS_MAX=640
    IMAGE_ROWS_MAX=360
    updGUI=QtCore.pyqtSignal()

    def __init__(self, parent=None):


        self.probabilidaes = 0
        self.etiquetas = 0


        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle("CamaraReal")
        self.setWindowIcon(QtGui.QIcon(path))
        self.resize(700,700)
        self.updGUI.connect(self.update)
        
        
        #figure
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        

        # Creo BoxLayout
        hbox = QtGui.QVBoxLayout()
     
        TestButton=QtGui.QPushButton("Evaluar")

        TestButton.clicked.connect(self.effect)

        self.imgLabel = QtGui.QLabel(self)

        self.img = None


        hbox.addWidget(self.imgLabel)
        hbox.addWidget(TestButton)
        hbox.addWidget(self.canvas)

        self.setLayout(hbox) 



    def setControl(self,control):
        self.control=control

    def update(self):
        #print 'updgui'

        self.probabilidaes = self.control.getProb()
        self.etiquetas = self.control.getLabels()
        


        image = self.control.getImage()
        if image != None:
            img = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
            size=QtCore.QSize(image.shape[1],image.shape[0])
            #self.imgLabel.resize(size)
            #self.resize(size)
            self.imgLabel.setPixmap(QtGui.QPixmap.fromImage(img))
            self.img = image
            #print 'printimg'
        if self.control.getFlag() == 1:


            self.plot()


            self.control.setFlag(0)



    def effect(self):
        #print 'pulsado'
        self.control.effect()

    def plot(self):

        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.hold(False)

        # plot data
        
        plt.bar(range(5),self.probabilidaes,align='center')
        plt.xticks(range(5), self.etiquetas,fontsize=20)
        plt.title('Clasification',fontsize=20)
        #plt.xlabel('Category',fontsize=20)
        plt.ylabel('Probability',fontsize=20)

        # refresh canvas
        self.canvas.draw()
