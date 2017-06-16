import cv2
import numpy as np


image =cv2.imread('flow.PNG')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(np.shape(hsv))


cv2.imwrite('retall1.png',hsv[:,0:1250,:])
cv2.imwrite('retall2.png',hsv[:,1750:3000,:])