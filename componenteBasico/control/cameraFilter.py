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
import numpy as np
import threading
from parallelIce.cameraClient import CameraClient

class CameraFilter:

    def __init__(self, camera):

        self.lock = threading.Lock()
        # MINE VARIABLES
        self.effectON = False
        self.flag = 0



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
        self.image = self.client.getImage()
        self.lock.release()        

    
    def getImage(self):
        '''
        self.lock.acquire()
        img = self.client.getImage()
        self.lock.release()
        '''
        img = self.image
        if self.effectON:
            img = self.opencvtest(img)

        return img



    def getFlag(self):

        return self.flag
    def setFlag(self,flag):
        self.flag = flag

    def effect(self):
        self.effectON = not self.effectON
        #print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")


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