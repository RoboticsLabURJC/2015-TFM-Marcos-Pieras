import cv2
import numpy as np


image1 =cv2.imread('scale1.png')
image2 =cv2.imread('scale2.png')


feature_params = dict( maxCorners = 60,qualityLevel = 0.15,minDistance = 2,blockSize = 7 )

old_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
equ = cv2.equalizeHist(old_gray)
p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)
corners = np.int0(p0)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(image2,(x,y),5,(0,0,255),-1)

cv2.imwrite('scale2points.png',image2)