import cv2
import numpy as np


image =cv2.imread('csn2.PNG')
#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(np.shape(image))


cv2.imwrite('retall1.png',image[:,0:1150,:])
cv2.imwrite('retall2.png',image[:,1350:2500,:])
