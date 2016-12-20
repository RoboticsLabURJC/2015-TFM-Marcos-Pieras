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
import os
import numpy as np
import threading
from parallelIce.cameraClient import CameraClient
import cv2
import sys
sys.path.insert(0, '/home/marc/Dropbox/tfmDeepLearning/semana4/boundingBox')
from claseXML import * 

class CameraFilter:

    def __init__(self, camera):

        self.pathGT = '/home/marc/Escriptori/ssdCom/groudnth'
        self.pathImages = '/home/marc/Escriptori/ssdCom/images'
        self.pathDetections = '/home/marc/Escriptori/ssdCom/deetctions'

        dirsGT = os.listdir( self.pathGT )
        self.dirsImages = os.listdir( self.pathImages )
        dirsDetections = os.listdir( self.pathDetections )
        self.numFiles = np.shape(dirsGT)[0]

        self.idxImage = 0
        image = cv2.imread(self.pathImages+'/'+self.dirsImages[self.idxImage])
        self.image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.lock = threading.Lock()
        # MINE VARIABLES
        self.effectON = False
        self.flag = 0

        self.flagNext = 0
        self.flagPrevious = 0


        self.client = camera

        self.height= self.client.getHeight()
        self.width = self.client.getWidth()
        self.image = 0

        if self.client.hasproxy():
            self.trackImage = np.zeros((self.height, self.width,3), np.uint8)
            self.trackImage.shape = self.height, self.width, 3

            self.thresoldImage = np.zeros((self.height,self. width,1), np.uint8)
            self.thresoldImage.shape = self.height, self.width,

    

    def update(self):
        self.lock.acquire()
        #self.image = self.client.getImage()
        self.image = self.getImage()

        self.lock.release()        
        
    
    def getImage(self):
        '''
        self.lock.acquire()
        img = self.client.getImage()
        self.lock.release()
        '''

        img = self.image
        imga = np.zeros(np.shape(img))

        if self.effectON:

            if self.flagNext == 1:

                self.idxImage += 1
                self.flagNext = 0
            
            else:
            
                self.idxImage -= 1
                self.flagPrevious = 0

            if self.idxImage>self.numFiles-1: 
            
                self.idxImage = 0
            
            if self.idxImage<0:
            
                self.idxImage = self.numFiles
            
            #print('sss',self.idxImage)
            #img = self.image = cv2.imread(self.pathImages+'/'+self.dirsImages[self.idxImage])
            self.effectON = False
        imge = self.image = cv2.imread(self.pathImages+'/'+self.dirsImages[self.idxImage])
        img =cv2.cvtColor(imge,cv2.COLOR_BGR2RGB)
        output = img.copy()
        
        #alpha = 0.6
        #cv2.rectangle(img, (50, 50), (100, 100),(0, 0, 255), -1)
        #cv2.addWeighted(img, alpha, output, 1 - alpha,0, output)

        # GT
        name = self.dirsImages[self.idxImage]
        idxPoint = name.find('.')
        namePoint = name[0:idxPoint]
        
        matrix = leerXMLimageLabel2(self.pathGT+'/'+namePoint+'.xml')
        #print(matrix)
        numObjectGT = np.shape(matrix)[0]
        for i in range(0,numObjectGT):
           
            #cv2.rectangle(img, (int(matrix[i][1]), int(matrix[i][2])), (int(matrix[i][3]), int(matrix[i][4])),(0, 255, 0), -1)
            cv2.rectangle(img, (int(matrix[i][1]), int(matrix[i][2])), (int(matrix[i][3]), int(matrix[i][4])),(0, 255, 0),2)


        # detecttions

        matrizDetc=leerXMLpropio(self.pathDetections+'/'+namePoint+'.xml')
        

        numObjectDetec = np.shape(matrizDetc)[0]
        for i in range(0,numObjectDetec):
           
            #cv2.rectangle(img, (int(matrizDetc[i][1]), int(matrizDetc[i][2])), (int(matrizDetc[i][3]), int(matrizDetc[i][4])),(255, 0, 0), -1)
            cv2.rectangle(img, (int(matrizDetc[i][1]), int(matrizDetc[i][2])), (int(matrizDetc[i][3]), int(matrizDetc[i][4])),(255, 0, 0),2)


        alpha = 0.6
        #cv2.addWeighted(img, alpha, output, 1 - alpha,0, output)

        return img
        


    def getFlag(self):

        return self.flag

    def setFlag(self,flag):
        self.flag = flag

    def effect(self,variable):
        #self.effectON = not self.effectON
        self.effectON = True
        
        #print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
        if variable == 1:

            self.flagPrevious = 1

        self.flagNext = 1

    def opencvtest(self,image):

        print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")


        self.flag = 1
        self.effectON = False
        return image


    def getColorImage(self):
        if self.client.hasproxy():
            self.lock.acquire()
            img = np.zeros((self.height, self.width,3), np.uint8)
            img = self.trackImage
            img.shape = self.trackImage.shape
            self.lock.release()
            return img
        return None

    def setColorImage(self,image):
        if self.client.hasproxy():
            self.lock.acquire()
            self.trackImage = image
            self.trackImage.shape = image.shape
            self.lock.release()

    def getThresoldImage(self):
        if self.client.hasproxy():
            self.lock.acquire()
            img = np.zeros((self.height, self.width,1), np.uint8)
            img = self.thresoldImage
            img.shape = self.thresoldImage.shape
            self.lock.release()
            return img
        return None

    def setThresoldImage(self,image):
        if self.client.hasproxy():
            self.lock.acquire()
            self.thresoldImage = image
            self.thresoldImage.shape = image.shape
            self.lock.release()