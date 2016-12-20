import sys, traceback, Ice
import jderobot
import numpy as np
import threading
import cv2
import time
#from matplotlib import pyplot as plt
#sys.path.insert(0, '/home/marc/caffe/python')
import caffe
from datetime import datetime

class Control():

    def __init__(self):
        self.lock = threading.Lock()
        self.effectON = False


        #model_file = '/home/marc/caffe/examples/mnist/lenet.prototxt'
        #pretrained_file = '/home/marc/caffe/examples/mnist/lenet_iter_10000.caffemodel'
	model_file = '/home/marc/caffe/models/VGGNet/coco/SSD_300x300/deploy.prototxt'
	pretrained_file = '/home/marc/caffe/models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_240000.caffemodel'
        self.xarxa = net = caffe.Classifier(model_file, pretrained_file, image_dims=(28, 28), raw_scale=255)



        try:
            ic = Ice.initialize(sys.argv)
            properties = ic.getProperties()
            camera = ic.propertyToProxy("Camarareal.Camera.Proxy")
            self.cameraProxy = jderobot.CameraPrx.checkedCast(camera)
            if self.cameraProxy:
                self.image = self.cameraProxy.getImageData("RGB8")
                self.height= self.image.description.height
                self.width = self.image.description.width
            else:
                print 'Interface camera not connected'

        except:
            traceback.print_exc()
            exit()
            status = 1

    def update(self):
        if self.cameraProxy:
            self.lock.acquire()
            #print 'updtcontrol'
            self.image = self.cameraProxy.getImageData("RGB8")
            self.height= self.image.description.height
            self.width = self.image.description.width
            self.lock.release()

    def getImage(self):
        if self.cameraProxy:
            self.lock.acquire()
            #print 'getimage'
            image = np.zeros((self.height, self.width, 3), np.uint8)
            image = np.frombuffer(self.image.pixelData, dtype=np.uint8)
            image.shape = self.height, self.width, 3
            if self.effectON:
                
                image = self.opencvtest(image)
                
                
            self.lock.release()
            return image

    def effect(self):
        self.effectON = not self.effectON

    def opencvtest(self, img):


        start_time = datetime.now()
        # Gaussian Filter
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(img,-1,kernel)

        # Edges
        edges = cv2.Canny(dst,100,200)
        # Dilatation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(18,18))
        dilation = cv2.dilate(edges,kernel,iterations = 1)

        # Contornos
        contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:


            x,y,w,h = cv2.boundingRect(c)
            aspect_ratio = float(w)/h


            #if aspect_ratio<0.98 and w>50:

            if aspect_ratio<0.98:


                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                imagenDigito = img[y:y+h,x:x+w]
                imagenDigitoGris = cv2.cvtColor(imagenDigito, cv2.COLOR_BGR2GRAY)
                ret,roi = cv2.threshold(imagenDigitoGris,200,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
                paddding = 10
                constant= cv2.copyMakeBorder(roi,paddding,paddding,paddding,paddding,cv2.BORDER_CONSTANT,value=0)


                imagef= cv2.resize(constant,(28,28))
                #imagef=cv2.normalize(np.float32(imagef),imagef, -1, 1, cv2.NORM_MINMAX)
                imagef=cv2.normalize(np.float32(imagef),imagef, 0, 1, cv2.NORM_MINMAX)
                self.xarxa.blobs['data'].reshape(1,1,28,28)

                self.xarxa.blobs['data'].data[...]=imagef


                output = self.xarxa.forward()

                digito = output['prob'].argmax()

                cv2.putText(img,str(digito), (x,y-4), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        end_time = datetime.now()
        dt = end_time - start_time
        print float(dt.microseconds)/1000
	self.effectON = False
        return img
