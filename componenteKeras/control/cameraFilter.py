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
sys.path.insert(0, '/home/marc/Dropbox/tfmDeepLearning/semana5/kltbasico')
from classFlow import * 
import time



class CameraFilter:

    def __init__(self, camera):


        self.pathSave = '/home/marc/Dropbox/tfmDeepLearning/semana5/componenteKLTreal/dataAssociation/record'
        # managment images database

        self.pathImages = '/home/marc/Escriptori/senora'
        self.dirsImages = os.listdir( self.pathImages )
        self.dirsImages.sort()
        self.numFiles = np.shape(self.dirsImages)[0]
        self.idxImage = 1
        image = cv2.imread(self.pathImages+'/'+self.dirsImages[self.idxImage])
        self.image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.lock = threading.Lock()

        self.numObjectDetec = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # Flags
        self.effectON = False
        self.flag = 0

        self.flagNext = 0
        self.flagPrevious = 0

        self.FrameNetwork = 50
        # previousImage
        self.previousImage = 0

        self.fls = 1 
        # deal detecttion
        self.flagDetection = 0
        self.detecciones = 0
        self.numDetection = 0

        self.FlagPulse = 1
        # managment KLT 
        self.det = 0
        
        self.varIndex = 0
        self.roi = np.zeros((3,4))
        self.roiBuffer = 0

        image = cv2.imread('/home/marc/Dropbox/tfmDeepLearning/semana5/componentesREALES/componenteKeras/logoJdeRobot.png',1) 
        
        self.img =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.alf = 1
        self.NormalFlow2 = 1
        
        self.NormalFlow = 1

        # count
        #self.listDetecciones = [51,77,104,125,150,175,200,225,250,275,300]
        self.listDetecciones = []
        self.detecCount = 0
        self.punteroDete = []
    def update(self):

        self.lock.acquire()


        if self.NormalFlow==1 and self.net.getFlagFinished()==0:

            if self.NormalFlow2 == 1:
                name = self.dirsImages[self.idxImage-1]
                idxPoint = name.find('.')
                namePoint = name[0:idxPoint]           
                self.net.processImage(namePoint)
                self.NormalFlow2 = 0
            
            img = self.img 
            
            while True:
                print('inside')
                self.roi = self.net.getDetections()
                self.roiBuffer = self.roi
                print('buf',self.roiBuffer)
                if self.net.getFlagFinished() == 1:
                    break


            self.punteroDete.append(np.shape(self.roi)[0])
            self.punterCUMSUM = np.cumsum(self.punteroDete)
            #print(np.shape(self.roi))
            img = self.img           
            self.image = img
            
        else:
            
        # We have the first detection and starts the normal flow, sends image 100 to net and starst LK
            startA = time.time()
            self.NormalFlow = 0
            
            if self.net.getFlagFinished() == 1:

                self.net.setFlagFinished()
                #self.listDetecciones.append(self.idxImage+self.FrameNetwork)
                print('sssssssssssssssss',self.idxImage+self.FrameNetwork)
                name = self.dirsImages[self.idxImage+self.FrameNetwork]
                self.listDetecciones.append(self.idxImage+self.FrameNetwork)
                idxPoint = name.find('.')
                namePoint = name[0:idxPoint]
                self.net.processImage(namePoint)
                
                #self.roi = self.net.getDetections()


                alfa = self.net.getDetections()

                if self.idxImage >10:
                    self.punteroDete.append(np.shape(alfa)[0])
                    self.punterCUMSUM = np.cumsum(self.punteroDete)
                    #print(self.punteroDete)
                    #print(np.shape(alfa))
                    self.roiBuffer = np.append(self.roiBuffer,alfa,0)
                    print('buf',self.roiBuffer)

            self.idxImage += 1
            
            # manage detections
            print(self.idxImage,self.listDetecciones[self.detecCount])
            if self.idxImage == self.listDetecciones[self.detecCount]:
                self.alf = 0
                print('cambio',self.detecCount,np.shape(self.roiBuffer))
                print(self.punteroDete,self.punterCUMSUM)
                print(self.punteroDete[self.detecCount],self.punteroDete[self.detecCount]+self.punteroDete[self.detecCount+1])
                
                #self.roi = self.roiBuffer[self.punteroDete[self.detecCount]:self.punteroDete[self.detecCount]+self.punteroDete[self.detecCount+1],0:4]
                self.roi = self.roiBuffer[self.punterCUMSUM[self.detecCount]:self.punterCUMSUM[self.detecCount+1],0:4]
                print(self.roi)
                self.detecCount +=1


            self.effectON = False
            
            frame1 = cv2.imread(self.pathImages+'/'+self.dirsImages[self.idxImage-1])
            frame2 = cv2.imread(self.pathImages+'/'+self.dirsImages[self.idxImage])
            frame2 =cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)

            img = frame2
            
            self.roi = np.array(self.roi)
            #self.roi2 = np.copy(self.roi)
            numObj= np.shape(self.roi)[0]
            listOf = []

            #print('num',np.shape(self.roi))
            for i in range(0,numObj): 

                roiFrame1 = frame1[int(self.roi[i,1]):int(self.roi[i,3]),int(self.roi[i,0]):int(self.roi[i,2])]
                roiFrame2 = frame2[int(self.roi[i,1]):int(self.roi[i,3]),int(self.roi[i,0]):int(self.roi[i,2])]
                self.roi[i,0],self.roi[i,1],self.roi[i,2],self.roi[i,3],lostTracklet = lucasKanadeTrackerMedianScaleStatic2(roiFrame1,roiFrame2,int(self.roi[i,0]),int(self.roi[i,1]),int(self.roi[i,2]),int(self.roi[i,3]))          
               
                
                if lostTracklet ==1:

                    listOf.append(i)
                    
                else:              

                    cv2.rectangle(img,(int(self.roi[i,0]),int(self.roi[i,1])),(int(self.roi[i,2]),int(self.roi[i,3])),(0,255,0),2)
                    
            

            self.roi = np.delete(self.roi, listOf, 0)
            

            endA = time.time()
            timeFPS = int(np.around(1/(endA-startA)))
            cv2.putText(img,str(timeFPS)+'-'+str(self.idxImage),(10,80), self.font, 3,(255,0,0),5,cv2.LINE_AA)
            
        n = str(self.idxImage)
        n2 = n.zfill(3)
        cv2.imwrite('/home/marc/Escriptori/comppnn'+'/'+n2+'.jpg',img)
        self.image = img

        self.lock.release()        
    
    def setNet(self,control):
        
        self.net = control

    def getImage(self):


        # Initizalization, sends first frame to the NET

        
            
        return self.image
        

    def getFlag(self):

        return self.flag

    def setFlag(self,flag):
        self.flag = flag

    def effect(self,variable):
        #self.effectON = not self.effectON
        self.effectON = True
        
        
        if float(variable) == 1.0:

            self.flagPrevious = 1
            self.flagNext = 0
            print('-----------------------P')
        else:
            self.flagPrevious = 0
            self.flagNext = 1
            print('-----------------------N')
    def detectGUI(self,unit):

        self.flagDetection = 1





    def opencvtest(self,image):

        print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")


        self.flag = 1
        self.effectON = False
        return image

    
