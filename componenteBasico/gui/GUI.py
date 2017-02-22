#
#  Copyright (C) 1997-2016 JDE Developers Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/.
#  Authors :
#       Alberto Martin Florido <almartinflorido@gmail.com>
#       Aitor Martinez Fernandez <aitor.martinez.fernandez@gmail.com>
#

from PyQt5.QtGui import QIcon,QImage,QPixmap
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow,QPushButton,QLabel,QHBoxLayout,QVBoxLayout,QWidget
from gui.communicator import Communicator
import sys
import os.path as osp
path = osp.join(osp.dirname(sys.modules[__name__].__file__), 'logoJdeRobot.png')
import cv2
import numpy as np

class Gui(QWidget):

    updGUI = pyqtSignal()

    def __init__(self, parent=None):

        
        super().__init__()

        self.setGeometry(300, 300, 300, 500)
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon(path))  


        # BUTTON
        TestButton=QPushButton("Evaluar")
        TestButton.clicked.connect(self.effect)       
        

        # AUXILIAR IMAGE
        self.imgLabel = QLabel(self)
        image = cv2.imread('logoJdeRobot.png',1)       
        image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        imgDef = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.imgLabel.setPixmap(QPixmap.fromImage(imgDef))

        # IMAGE PRINCIPAL
        self.imgPrincipal = QLabel(self)

        # Configuracion BOX
        vbox = QVBoxLayout()
        vbox.addWidget(self.imgPrincipal)
        vbox.addWidget(self.imgLabel)
        vbox.addWidget(TestButton)
        self.setLayout(vbox) 

        # Necesario
        self.cameraCommunicator = Communicator()
        self.trackingCommunicator = Communicator()
        self.show()

    def setControl(self,control):

    	self.control = control

    def update(self):

    	input_image = self.control.getImage()
    	#print (np.shape(input_image))
    	imgDef = QImage(input_image.data, input_image.shape[1], input_image.shape[0], QImage.Format_RGB888)
    	self.imgPrincipal.setPixmap(QPixmap.fromImage(imgDef))
    	
    	if self.control.getFlag()==1:


    		self.imgLabel.setPixmap(QPixmap.fromImage(imgDef))
    		self.control.setFlag(0)

        #print ("nigga")

    def effect(self):
    	self.control.effect()
        #print("----------------------------pm")