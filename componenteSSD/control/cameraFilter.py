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
import sys
import time
import matplotlib.pyplot as plt
#sys.path.insert(0, '/home/marc/caffe/python')
import caffe
import cv2

class CameraFilter:

    def __init__(self, camera):

        self.lock = threading.Lock()
        # MINE VARIABLES
        self.effectON = False
        self.flag = 0
        caffe.set_mode_cpu()

        # tema caffe
        model_file = '/home/marc/coco/deploy.prototxt'
        pretrained_file = '/home/marc/coco/VGG_coco_SSD_300x300_iter_240000.caffemodel'      
        
       
        self.net = caffe.Net(model_file,pretrained_file,caffe.TEST)
        file = open('/home/marc/caffe/data/coco/labels.txt', 'r')
        self.lines=file.readlines() 
		
        # fin caffe
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

    def get_label2(self,lines,v):
        labelnames = []
    

        for i in v:
            #labelnames.append(lines[i][4:])
            labelnames.append(lines[int(i-1)][4:])
    
        return labelnames

    def getFlag(self):

        return self.flag
    def setFlag(self,flag):
        self.flag = flag

    def effect(self):
        self.effectON = not self.effectON
        #print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")


    def opencvtest(self,img):

        print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")


        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104,117,123])) # mean pixel


        # set net to batch size of 1
        image_resize = 300
        self.net.blobs['data'].reshape(1,3,image_resize,image_resize)

        transformed_image = transformer.preprocess('data', img)

        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        startA = time.time()
        detections = self.net.forward()['detection_out']
        endA = time.time()
        
        print ('Executed in', str((endA - startA)*1000))

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = self.get_label2(self.lines,top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        print(top_labels)


        coco = 81
        colors = plt.cm.hsv(np.linspace(0, 1, coco)).tolist()


        currentAxis = plt.gca()

        font = cv2.FONT_HERSHEY_SIMPLEX

        #for i in xrange(top_conf.shape[0]):
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            display_txt = '%s: %.2f'%(label_name, score)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            
            colorCV = 256*color
            
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
            #cv2.putText(img,'Person',(xmin,ymin), font, 0.7,(255,255,255),2)
            cv2.putText(img,label_name,(xmin-5,ymin-5), font, 1,(255,0,0),2)










        #self.outputData = np.array([top_conf,top_label_indices,top_labels,top_xmin,top_ymin,top_xmax,top_ymax])     

        self.flag = 1
        self.effectON = False
        return img


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