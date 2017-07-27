import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pltind


'''
performance = np.array([28.2,25.8,16.4,11.7,6.7,3.57])


objects = ('2010', '2011', '2012', '2013', '2014', '2015')
y_pos = np.arange(len(objects))
#performance = [10,8,6,4,2,1]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects,fontsize=30)
plt.ylabel('Top-5 error',fontsize=40)
plt.xlabel('Year',fontsize=40)
plt.title('ILSVRC',fontsize=60)
plt.tick_params(labelsize=20)
plt.tick_params(labelsize=40)

plt.show()

'''
'''

performance = np.array([18,22,27,36,41,41.5,51,63,76])


objects = ('2007','2008','2009','2010', '2011', '2012', '2013', '2014', '2015')
y_pos = np.arange(len(objects))
#performance = [10,8,6,4,2,1]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects,fontsize=30)
plt.ylabel('Mean average precision',fontsize=40)
plt.xlabel('Year',fontsize=40)
plt.title('VOC07',fontsize=60)
plt.tick_params(labelsize=20)
 
plt.show()
'''

'''
import cv2
import numpy as np
img1a = cv2.imread('out11.jpg')
img2b = cv2.imread('out3.jpg')
img3b = cv2.imread('out9.jpg')

imgAux = cv2.imread('out11.jpg')


img1 = cv2.resize(img1a,(64, 128), interpolation = cv2.INTER_CUBIC)
img2 = cv2.resize(img2b,(64, 128), interpolation = cv2.INTER_CUBIC)
img3 = cv2.resize(img3b,(64, 128), interpolation = cv2.INTER_CUBIC)

imgA = cv2.resize(imgAux,(64, 128), interpolation = cv2.INTER_CUBIC)

res1 = cv2.rectangle(img1, (int(0),int(0)), (int(64),int(128)), (0,255,0), 5)

res2 = cv2.rectangle(img2, (int(0),int(0)), (int(64),int(128)), (0,255,0), 5)

res3 = cv2.rectangle(img3, (int(0),int(0)), (int(64),int(128)), (0,0,255), 5)

resA = cv2.rectangle(imgA, (int(0),int(0)), (int(64),int(128)), (0,0,255), 5)

vis1 = np.concatenate((res1, res2), axis=1)


vis2 = np.concatenate((resA, res3), axis=1)

vis3 = np.concatenate((vis1, vis2), axis=0)

cv2.imwrite('out2.png', vis3)



import matplotlib.pyplot as plt
import numpy.random as rnd

fig = plt.figure()
plt.subplot(221)
plt.imshow(img1)
plt.subplot(222)
plt.imshow(img2)
plt.subplot(223)
plt.imshow(img1)
plt.subplot(224)
plt.imshow(img3)

fig.tight_layout()
plt.show()

'''


#performance = np.array([18,22,27,36,41,41.5,51,63,76])

performance1 = np.array([18,22,27,36,41,41.5,0,0,0])

performance2 = np.array([0,0,0,0,0,0,51,63,76])


objects = ('2007','2008','2009','2010', '2011', '2012', '2013', '2014', '2015')
y_pos = np.arange(len(objects))
#performance = [10,8,6,4,2,1]
 
plt.bar(y_pos, performance2, align='center', alpha=0.5)
plt.bar(y_pos, performance1, align='center', alpha=0.5)
plt.xticks(y_pos, objects,fontsize=30)
plt.ylabel('Mean average precision',fontsize=40)
plt.xlabel('Year',fontsize=40)
plt.title('VOC07',fontsize=60)
plt.tick_params(labelsize=40)

 
plt.show()





performance = np.array([0.0,0.0,16.4,11.7,6.7,3.57])


performance2 = np.array([28.2,25.8,0,0,0,0])


objects = ('2010', '2011', '2012', '2013', '2014', '2015')
y_pos = np.arange(len(objects))
#performance = [10,8,6,4,2,1]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.bar(y_pos, performance2, align='center', alpha=0.5)
plt.xticks(y_pos, objects,fontsize=30)
plt.ylabel('Top-5 error',fontsize=40)
plt.xlabel('Year',fontsize=40)
#plt.title('ILSVRC',fontsize=60)
plt.tick_params(labelsize=40)
 
plt.show()
