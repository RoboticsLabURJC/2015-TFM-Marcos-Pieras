import sys, traceback, Ice
import jderobot
import numpy as np
import threading
import cv2
from matplotlib import pyplot as plt
sys.path.insert(0, '/home/marc/caffe/python')
import caffe
from control.cameraFilter import CameraFilter


class Control():

    def __init__(self,camera):
        self.lock = threading.Lock()
        self.effectON = False
        self.cameraAtribute = camera

    def update(self):

        input_image = self.cameraAtribute.getImage()
        #print (np.shape(input_image))
        print("control")

    def getS(self):

        return self.cameraAtribute 
 
    def effect(self):
        self.effectON = not self.effectON

    def opencvtest(self, img):

        return img
