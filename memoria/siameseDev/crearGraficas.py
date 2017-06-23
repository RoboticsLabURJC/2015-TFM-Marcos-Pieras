import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pltind


import cv2
import numpy as np

'''
img1b = cv2.imread('appe/lowTexture/townCenter_000026_0168.0.jpg')
img2b = cv2.imread('appe/lowTexture/townCenter_000026_0169.0.jpg')
img3b = cv2.imread('appe/lowTexture/townCenter_000026_0170.0.jpg')
img4b = cv2.imread('appe/lowTexture/townCenter_000026_0171.0.jpg')
img5b = cv2.imread('appe/lowTexture/townCenter_000026_0172.0.jpg')
img6b = cv2.imread('appe/lowTexture/townCenter_000026_0173.0.jpg')
'''

img1b = cv2.imread('flatten.jpg')
img2b = cv2.imread('globalPooling.png')
img3b = cv2.imread('spp.png')
#img4b = cv2.imread('foto156.png')
#img5b = cv2.imread('foto158.png')



'''
feature_params = dict( maxCorners = 60,qualityLevel = 0.15,minDistance = 2,blockSize = 7 )

old_gray1 = cv2.cvtColor(img1b, cv2.COLOR_RGB2GRAY)
equ1 = cv2.equalizeHist(old_gray1)
p01 = cv2.goodFeaturesToTrack(equ1, mask = None, **feature_params)

old_gray2 = cv2.cvtColor(img2b, cv2.COLOR_RGB2GRAY)
equ2 = cv2.equalizeHist(old_gray2)
p02 = cv2.goodFeaturesToTrack(equ2, mask = None, **feature_params)

old_gray3 = cv2.cvtColor(img3b, cv2.COLOR_RGB2GRAY)
equ3 = cv2.equalizeHist(old_gray3)
p03 = cv2.goodFeaturesToTrack(equ3, mask = None, **feature_params)

old_gray4 = cv2.cvtColor(img4b, cv2.COLOR_RGB2GRAY)
equ4 = cv2.equalizeHist(old_gray4)
p04 = cv2.goodFeaturesToTrack(equ4, mask = None, **feature_params)

old_gray5 = cv2.cvtColor(img5b, cv2.COLOR_RGB2GRAY)
equ5 = cv2.equalizeHist(old_gray5)
p05 = cv2.goodFeaturesToTrack(equ5, mask = None, **feature_params)

old_gray6 = cv2.cvtColor(img6b, cv2.COLOR_RGB2GRAY)
equ6 = cv2.equalizeHist(old_gray6)
p06= cv2.goodFeaturesToTrack(equ6, mask = None, **feature_params)


numPoints1 = np.shape(p01)[0]
numPoints2 = np.shape(p02)[0]
numPoints3 = np.shape(p03)[0]
numPoints4 = np.shape(p04)[0]
numPoints5 = np.shape(p05)[0]
numPoints6 = np.shape(p06)[0]

print(np.shape(p01))
for item in range(0,numPoints1):
	x = p01[item,0,0]
	y = p01[item,0,1]
	cv2.circle(img1b, (x,y), 3, (0,0,255), -1)


for item in range(0,numPoints2):
	x = p02[item,0,0]
	y = p02[item,0,1]
	cv2.circle(img2b, (x,y), 3, (0,0,255), -1)

for item in range(0,numPoints3):
	x = p03[item,0,0]
	y = p03[item,0,1]
	cv2.circle(img3b, (x,y), 3, (0,0,255), -1)

for item in range(0,numPoints4):
	x = p04[item,0,0]
	y = p04[item,0,1]
	cv2.circle(img4b, (x,y), 3, (0,0,255), -1)

for item in range(0,numPoints5):
	x = p05[item,0,0]
	y = p05[item,0,1]
	cv2.circle(img5b, (x,y), 3, (0,0,255), -1)

for item in range(0,numPoints6):
	x = p06[item,0,0]
	y = p06[item,0,1]
	cv2.circle(img6b, (x,y), 3, (0,0,255), -1)

'''
img1 = cv2.resize(img1b,(777,573), interpolation = cv2.INTER_CUBIC)
img2 = cv2.resize(img2b,(777,573), interpolation = cv2.INTER_CUBIC)
img3 = cv2.resize(img3b,(777,573), interpolation = cv2.INTER_CUBIC)
#img4 = cv2.resize(img4b,(336, 486), interpolation = cv2.INTER_CUBIC)
#img5 = cv2.resize(img5b,(336, 486), interpolation = cv2.INTER_CUBIC)




#vis1 = np.concatenate((img1, img2,img3,img4,img5), axis=1)



cv2.imwrite('flatten2.png', img1)
cv2.imwrite('global2.png', img2)
cv2.imwrite('spp2.png', img3)



