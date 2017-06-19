import cv2
import numpy as np
import os
import time
import sys
sys.path.insert(0, '/home/marc/Dropbox/tfmDeepLearning/semana6/mejoraLK')
from classFlow import *
import time
import matplotlib.pyplot as plt


# keras
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import sys
sys.path.append('/home/marc/SSD-Tensorflow/')


from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
import threading

DETECTION_RATE = 30

#PATH_IMAGES = '/home/marc/datasetTracking/mot16/train/MOT16-02/img1'
PATH_IMAGES = '/home/marc/Escriptori/senora'
PATH_SAVE = '/home/marc/Dropbox/tfmDeepLearning/semana7/componenteSSDtf/provisionalResult'

listImages = os.listdir( PATH_IMAGES )
listImages.sort()
numFiles = np.shape(listImages)

MAX_FRAME = numFiles[0]

listOfDetections = [i*DETECTION_RATE for i in range(0,1000) if i*DETECTION_RATE <= MAX_FRAME]
NUM_ITERATIONS = np.shape(listOfDetections)[0]

class myThread (threading.Thread):
        def __init__(self,NUM_DETCTIONS):
            threading.Thread.__init__(self)
 
            self.NUM = NUM_DETCTIONS
            self.isess = tf.InteractiveSession()
            net_shape = (300, 300)
            data_format = 'NHWC'
            self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
            image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            self.image_4d = tf.expand_dims(image_pre, 0)
            reuse = None
            ssd_net = ssd_vgg_300.SSDNet()
            with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
                self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=reuse)

            ckpt_filename = '/home/marc/SSD-Tensorflow/checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'

            self.isess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.isess, ckpt_filename)
            self.ssd_anchors = ssd_net.anchors(net_shape)
            self.detecction = []
            self.iDetections = []

        def resultadso(self):
            return self.detecction
        def iniResult(self):
            self.detecction = []

        def run(self):
            


            for i in range(0,self.NUM):
 
                startA = time.time()

                image_names = sorted(os.listdir(PATH_IMAGES))

                indexFrame = listOfDetections[i]

                print('sadsdasd',indexFrame)
                img = cv2.imread(PATH_IMAGES+'/'+image_names[indexFrame],1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                select_threshold=0.2
                nms_threshold=.45
                net_shape=(300, 300)


                #def process_image(img, select_threshold=0.2, nms_threshold=.45, net_shape=(300, 300)):

                rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],feed_dict={self.img_input: img})


                # Get classes and bboxes from the net outputs.
                rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(rpredictions, rlocalisations, self.ssd_anchors,select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)


                idx = np.where(rclasses != 15)[0]
                rclasses = np.delete(rclasses, idx,0)   
                rscores = np.delete(rscores, idx,0)   
                rbboxes = np.delete(rbboxes, idx,0)    

                rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
                rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
                rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
                rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

                sizeImg = np.shape(img)
                height = sizeImg[0]
                width = sizeImg[1]

                sizeDetection = np.shape(rbboxes)
                
                detecctionNET = np.zeros(sizeDetection)

                ymin = (np.around(rbboxes[:, 0] * height))
                xmin = (np.around(rbboxes[:, 1] * width))
                ymax = (np.around(rbboxes[:, 2] * height))
                xmax = (np.around(rbboxes[:, 3] * width))


                detecctionNET[:,0]=xmin
                detecctionNET[:,1]=ymin
                detecctionNET[:,2]=xmax
                detecctionNET[:,3]=ymax

                #self.detecction = detecctionNET
                
                self.detecction.append(detecctionNET)
                #print(np.shape(self.detecction))
                #self.iDetections.append(sizeDetection[0])


                #print(rbboxes)
                end = time.time()
                print('timeRED',end-startA)


thread= myThread(NUM_ITERATIONS)
thread.start()



while(True):

    if thread.resultadso() != []:
        break



initialDetection = thread.resultadso()
print(thread.resultadso())

 


time.sleep(2)




'''
listImages = os.listdir( PATH_IMAGES )
listImages.sort()
numFiles = np.shape(listImages)
'''


roi1 = initialDetection[0][:][:]


roi = np.copy(roi1)
numPersonas = np.shape(roi)[0]

MAX_PERSON = 500
colors = np.random.randint(0,255,(MAX_PERSON,3))

randomIdentities = np.random.randint(1,255,(MAX_PERSON,1))
roiID = randomIdentities[:numPersonas]
roiColors = colors[:numPersonas]


colors = np.delete(colors, np.s_[:numPersonas], 0)
randomIdentities = np.delete(randomIdentities, np.s_[:numPersonas], 0)


#velocidades = np.zeros((16,2))
velocidades = np.zeros((numPersonas,2))
flagVelocidades = 0

detectionestemps = []
detectiones = []
sats = np.zeros([numFiles[0]-1,3])




listOfDetections.append(10000)
indexDetections = 1



results = []

#frame1x = cv2.imread(PATH_IMAGES+'/'+listImages[0],0)
frame1x = cv2.imread(PATH_IMAGES+'/'+listImages[0])

for x in range(1,numFiles[0]-1):

    
    startA = time.time()

    if x == 1:
        
        frame1 = frame1x


    
    #frame2 = cv2.imread(PATH_IMAGES+'/'+listImages[x],0)
    frame2 = cv2.imread(PATH_IMAGES+'/'+listImages[x])
    framePintar = np.copy(frame2)
    endImage = time.time()
    sats[x,0]= endImage-startA
    startB = time.time()
    
    # dataAssociation
    print(x,listOfDetections[indexDetections],indexDetections)
    
    if x == listOfDetections[indexDetections]:
        



        print('s',listOfDetections[indexDetections],indexDetections)
        detecctionBuff = thread.resultadso()
        print(np.shape(detecctionBuff))
        
        #print(listIndex,listIndex[indexDetections],np.shape(detecction))
        #print(detecctionBuff[indexDetections][:][:])
        detecction = detecctionBuff[indexDetections][:][:]
        print('numD',np.shape(detecction))

        
        roiAux = []
        roiColorsAux = []
        roiIDaux = []

        lukasCentreX  = roi[:,0]+((roi[:,2]-roi[:,0])/2)
        lukasCentreY  = roi[:,1]+((roi[:,3]-roi[:,1])/2)

        detCentreX  = detecction[:,0]+((detecction[:,2]-detecction[:,0])/2)
        detCentreY  = detecction[:,1]+((detecction[:,3]-detecction[:,1])/2)

        numLK = np.shape(roi)
        print('numLK',numLK)
        for idx in range(0,numLK[0]):

            numD = np.shape(detecction)[0]

            if numD == 0:

                roiAux.append(roi[idx,:])
                #roi[idx,:]=roi[idx,:]
                roiColorsAux.append(roiColors[idx,:])
                roiIDaux.append(roiID[idx,:])
                continue

            distancia = np.sqrt( (lukasCentreX[idx]-detCentreX)**2 + (lukasCentreY[idx]-detCentreY)**2 )
            print(distancia)

            numD = np.shape(detecction)

            idxMinDistance = np.argmin(distancia)

            if distancia[idxMinDistance] < 40.0:

                roiAux.append(detecction[idxMinDistance,:])
                roiColorsAux.append(roiColors[idx,:])
                roiIDaux.append(roiID[idx,:])

                detCentreX = np.delete(detCentreX,idxMinDistance)
                detCentreY = np.delete(detCentreY,idxMinDistance)
                detecction = np.delete(detecction,idxMinDistance,0)

            else:

                roiAux.append(roi[idx,:])
                #roi[idx,:]=roi[idx,:]
                roiColorsAux.append(roiColors[idx,:])
                roiIDaux.append(roiID[idx,:])
                
        
        numDeteccionesSin = np.shape(detecction)
        #print('numDeteccionsSNES',numDeteccionesSin)
        for xs in range(0,numDeteccionesSin[0]):

            roiAux.append(detecction[xs,:])
            roiColorsAux.append(colors[xs,:])
            roiIDaux.append(randomIdentities[xs,:])

            randomIdentities = np.delete(randomIdentities, xs, 0)
            colors = np.delete(colors, xs, 0)

        roi = np.copy(roiAux)
        roiColors = np.copy(roiColorsAux)
        roiID = np.copy(roiIDaux)

        numPersonas = np.shape(roi)[0]
        #roiColors = colors[:numPersonas]
        velocidades = np.zeros((numPersonas,2))
        #print('change',indexDetections,listImages[x])
        
        indexDetections += 1
        flagVelocidades =1
        #print(thread.resultadso())

        time.sleep(0.5)
    
    
    listOf = []
    
    

    for iPerson in range(0,numPersonas):

        roiFrame1 = frame1[int(roi[iPerson,1]):int(roi[iPerson,3]),int(roi[iPerson,0]):int(roi[iPerson,2])]
        roiFrame2 = frame2[int(roi[iPerson,1]):int(roi[iPerson,3]),int(roi[iPerson,0]):int(roi[iPerson,2])]

        roi[iPerson,0],roi[iPerson,1],roi[iPerson,2],roi[iPerson,3],tracas,dx,dy = lucasKanadeTrackerMedianScaleStatic2PlusOptimized2Deploy(roiFrame1,roiFrame2,roi[iPerson,0],roi[iPerson,1],roi[iPerson,2],roi[iPerson,3])
        #roi[iPerson,0],roi[iPerson,1],roi[iPerson,2],roi[iPerson,3],tracas,dx,dy = lucasKanadeTrackerMedianScaleStatic3Deploy(roiFrame1,roiFrame2,roi[iPerson,0],roi[iPerson,1],roi[iPerson,2],roi[iPerson,3],x)

        


        # modulo velocidad


        #if x==1 or x == 64  or x == 128 or x == 193 or x==258:
        if flagVelocidades==1:
            velocidades[iPerson,0]=dx
            velocidades[iPerson,1]=dy
            flagVelocidades = 0

        # X
        if np.absolute(velocidades[iPerson,0])==0.0:

            if np.absolute(velocidades[iPerson,1])==0.0:

                vxPorcTemporal = 0.0
                vyPorcTemporal = 0.0

            else:

                vxPorcTemporal = 0.0
                vyPorcTemporal = np.absolute(dy-velocidades[iPerson,1])/np.absolute(velocidades[iPerson,1])

        
        # Y 
        elif np.absolute(velocidades[iPerson,1])==0.0:

            if np.absolute(velocidades[iPerson,0])==0.0:

                vxPorcTemporal = 0.0
                vyPorcTemporal = 0.0
            else:


                vxPorcTemporal = np.absolute(dx-velocidades[iPerson,0])/np.absolute(velocidades[iPerson,0])
                vyPorcTemporal = 0.0
        else:
            
            vxPorcTemporal = np.absolute(dx-velocidades[iPerson,0])/(np.absolute(velocidades[iPerson,0]))
            vyPorcTemporal = np.absolute(dy-velocidades[iPerson,1])/(np.absolute(velocidades[iPerson,1]))

            
        velocidades[iPerson,0] = dx
        velocidades[iPerson,1] = dy

    
        n = str(x)
        n2 = n.zfill(3)

        aStr = str(iPerson)
        n21 = aStr.zfill(3)
        
        #print("{0},{1:3.2f},{2:3.2f},{3:3.2f},{4:3.2f},{5}".format(roiID[iPerson][0],dx,dy,vxPorcTemporal,vyPorcTemporal,tracas))
        #print("{0:4d},{1:3.2f},{2:3.2f},{3}".format(roiID[iPerson][0],vxPorcTemporal,vyPorcTemporal,tracas))


        #print(roiID[iPerson][0],dx,dy,vxPorcTemporal,vyPorcTemporal,tracas)
        #if (vxPorcTemporal >= 3.4 and vyPorcTemporal >= 3.4) or tracas == 1 or vyPorcTemporal > 11.0 or vxPorcTemporal > 11.0:
        limitBOTH = 10.0
        limitTemporal = 45.0
        if (vxPorcTemporal >= limitBOTH and vyPorcTemporal >=limitBOTH) or (tracas == 1) or (vyPorcTemporal > limitTemporal) or (vxPorcTemporal > limitTemporal):

        #if tracas == 1:
            
            print('mateu',x,roiID[iPerson][0],iPerson,vxPorcTemporal,tracas,vyPorcTemporal)
            listOf.append(iPerson)
        else:
            cv2.putText(framePintar,str(roiID[iPerson][0]),(int(roi[iPerson,0]),int(roi[iPerson,1])),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA) 
            cv2.rectangle(framePintar,(int(roi[iPerson,0]),int(roi[iPerson,1])),(int(roi[iPerson,2]),int(roi[iPerson,3])),(int(roiColors[iPerson,0]),int(roiColors[iPerson,1]),int(roiColors[iPerson,2])),3)
            #cv2.rectangle(framePintar,(int(roi[iPerson,0]),int(roi[iPerson,1])),(int(roi[iPerson,2]),int(roi[iPerson,3])),(255,0,255),3)
            results.append([int(x+1),int(roiID[iPerson][0]),int(roi[iPerson,0]+1),int(roi[iPerson,1]+1),int(roi[iPerson,2]-roi[iPerson,0]+1),int(roi[iPerson,3]-roi[iPerson,1]+1),-1,-1,-1,-1])



    endILK = time.time()
    sats[x,1]= endILK-startB

    startC = time.time()
    #print(x,numPersonas,sats[x,1])    
    frame1 = np.copy(frame2)

    roi = np.delete(roi, listOf,0)    
    roiColors = np.delete(roiColors, listOf,0) 
    roiID = np.delete(roiID,listOf,0) 

    timeFPS = int(np.around(1/(endILK-startA)))
    #cv2.putText(framePintar,str(timeFPS)+'-'+str(x),(10,80),cv2.FONT_HERSHEY_SIMPLEX, 3,(255,0,0),5,cv2.LINE_AA)   
    n = str(x)
    n2 = n.zfill(3)
    cv2.imwrite(PATH_SAVE+'/'+n2+'.jpg',framePintar)
    
    numPersonas = np.shape(roi)[0]  
    endC = time.time()
    #detectionestemps.append([x,endC-startA])
    
    sats[x,2]= endC-startC
    #print(x,sats[x,0])
    #print('FIN',numPersonas,np.shape(roi2))
    #print('----------------------------------------')



siz =  range(0,numFiles[0]-1)
print('ds',np.mean(sats[:,0]),np.mean(sats[:,1]),np.mean(sats[:,2]),np.mean(sats[:,0]+sats[:,1]+sats[:,2]),np.mean(sats[:,0]+sats[:,1]+sats[:,2])/numFiles[0])
'''
fig = plt.figure()

plt.bar(siz,sats[:,0]+sats[:,1]+sats[:,2], color='r')
plt.bar(siz,sats[:,0]+sats[:,1], color='b')
plt.bar(siz,sats[:,1], color='y')
plt.show()
'''
#np.savetxt('/home/marc/Dropbox/tfmDeepLearning/semana8/componente/MOT16-13.txt', results, delimiter=',',fmt="%d") 
