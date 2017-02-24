#
#  Copyright (C) 1997-2016 JDE Developers Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or

from PyQt5.QtGui import QIcon,QImage,QPixmap
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow,QPushButton,QLabel,QHBoxLayout,QVBoxLayout,QWidget
from gui.communicator import Communicator
import sys
import os.path as osp
path = osp.join(osp.dirname(sys.modules[__name__].__file__), 'logoJdeRobot.png')
import cv2
import numpy as np
import os
import time

class Gui(QWidget):

    updGUI = pyqtSignal()

    def __init__(self, parent=None):

        
        super().__init__()

        # Directories
        self.setGeometry(300, 300, 1200, 1000)
        self.setWindowTitle('Compare')
        self.setWindowIcon(QIcon(path))  

        # AUXILIAR IMAGE
        self.imgLabel = QLabel(self)
        
        image = cv2.imread('/home/marc/Dropbox/tfmDeepLearning/semana5/componentesREALES/componenteKeras/logoJdeRobot.png',1)   
        image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        imgDef = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.imgLabel.setPixmap(QPixmap.fromImage(imgDef))

         # Configuracion BOX
        vbox = QVBoxLayout()
        vbox.addWidget(self.imgLabel)

        hbox = QHBoxLayout()
        vbox.addLayout(hbox)
        

        self.setLayout(vbox) 

        # Necesario
        self.cameraCommunicator = Communicator()
        self.trackingCommunicator = Communicator()
        self.show()

    def setControl(self,control):

    	self.control = control

    def update(self):
        #startA = time.time()
        input_image = self.control.getImage()
        imgDef = QImage(input_image.data, input_image.shape[1], input_image.shape[0], QImage.Format_RGB888)
        scaledImage = imgDef.scaled(self.imgLabel.size())
        self.imgLabel.setPixmap(QPixmap.fromImage(scaledImage))
        #endA = time.time()
        #print(endA-startA)
        