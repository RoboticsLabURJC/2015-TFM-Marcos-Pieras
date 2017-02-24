#
#  Copyright (C) 1997-2016 JDE Developers Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or


import sys
import os.path as osp
import cv2
import numpy as np
import os
import time
import caffe
sys.path.insert(0, '/home/marc/Dropbox/tfmDeepLearning/semana5/componenteKLTDMultiple')
from claseXML import *

import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from scipy.misc import imread
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import os
os.chdir( "/home/marc/ssd_keras/" )
import sys
sys.path.append('/home/marc/ssd_keras/')
from ssd import SSD300
from ssd_utils import BBoxUtility
import time




class Net:

    #updGUI = pyqtSignal()

    def __init__(self, parent=None):


        # net definition
        # flags
        #   for start the process
        self.fStart = 0
        self.flagFinished = 0

        # auxiliary variables 
        self.image = 0
        self.LastDetections = 0
        self.pathDetections = '/home/marc/Escriptori/senotaXML'
        self.nameImage = 0

    def update(self):

        if self.fStart == 1:
            
            self.fStart = 0
            
            startA = time.time()

            voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
            NUM_CLASSES = len(voc_classes) + 1


            input_shape=(300, 300, 3)
            model = SSD300(input_shape, num_classes=NUM_CLASSES)
            model.load_weights('weights_SSD300.hdf5', by_name=True)
            bbox_util = BBoxUtility(NUM_CLASSES)

            inputs = []
            print(self.nameImage)
            imgO = cv2.imread('/home/marc/Escriptori/senora/'+self.nameImage+'.jpg',1)

            img =cv2.cvtColor(imgO,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (300, 300),interpolation = cv2.INTER_CUBIC) 
            inputs.append(img.copy())
            inputs = preprocess_input(np.array(inputs,dtype=np.float64))


            preds = model.predict(inputs, batch_size=1, verbose=1)
            results = bbox_util.detection_out(preds)

            tamImage = np.shape(imgO)

            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            det_xmin = results[0][:, 2]
            det_ymin = results[0][:, 3]
            det_xmax = results[0][:, 4]
            det_ymax = results[0][:, 5]

            idx =np.where(det_label==15.)
            idx2 = np.where(det_conf>=0.25)
            idxGLobal = list(set(idx[0])&set(idx2[0]))

            roi = np.zeros([np.shape(idxGLobal)[0],4])

            roi[:,0]=det_xmin[idxGLobal]
            roi[:,1]=det_ymin[idxGLobal]
            roi[:,2]=det_xmax[idxGLobal]
            roi[:,3]=det_ymax[idxGLobal]



            roi[:,0] =  np.around(roi[:,0]*tamImage[1])
            roi[:,1] =  np.around(roi[:,1]*tamImage[0])
            roi[:,2] =  np.around(roi[:,2]*tamImage[1])
            roi[:,3] =  np.around(roi[:,3]*tamImage[0])
            endA = time.time()
            #print('timeNet',endA-startA)
            
            self.LastDetections = roi
            self.flagFinished = 1
            

    def getFlagFinished(self):
        
        return self.flagFinished

    def setFlagFinished(self):
        self.flagFinished = 0 

    def setFlagFinishedOne(self):
        self.flagFinished = 1 
    def getDetections(self):
        return self.LastDetections

    def processImage(self,image):

        self.image = image
        self.fStart = 1
        self.nameImage = image
        '''
        if self.flagFinished == 1:
            return self.LastDetections
        return 0
        '''