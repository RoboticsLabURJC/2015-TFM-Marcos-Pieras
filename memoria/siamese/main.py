import cv2
import numpy as np


image =cv2.imread('ztottal.PNG')
#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(np.shape(image))


cv2.imwrite('retall1.png',image[:,0:800,:])
cv2.imwrite('retall2.png',image[:,800:1600,:])
cv2.imwrite('retall3.png',image[:,1600:2400,:])