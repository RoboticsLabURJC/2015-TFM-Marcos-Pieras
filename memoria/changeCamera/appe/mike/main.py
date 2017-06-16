import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pltind


import cv2
import numpy as np

img1b = cv2.imread('townCenter_000028_0109.0.jpg')
img2b = cv2.imread('townCenter_000028_0110.0.jpg')
img3b = cv2.imread('townCenter_000028_0111.0.jpg')
img4b = cv2.imread('townCenter_000028_0112.0.jpg')
img5b = cv2.imread('townCenter_000028_0113.0.jpg')
img6b = cv2.imread('townCenter_000028_0114.0.jpg')



img1 = cv2.resize(img1b,(344, 500), interpolation = cv2.INTER_CUBIC)
img2 = cv2.resize(img2b,(344, 500), interpolation = cv2.INTER_CUBIC)
img3 = cv2.resize(img3b,(344, 500), interpolation = cv2.INTER_CUBIC)
img4 = cv2.resize(img4b,(344, 500), interpolation = cv2.INTER_CUBIC)
img5 = cv2.resize(img5b,(344, 500), interpolation = cv2.INTER_CUBIC)
img6 = cv2.resize(img6b,(344, 500), interpolation = cv2.INTER_CUBIC)



vis1 = np.concatenate((img1, img2,img3,img4,img5,img6), axis=1)



cv2.imwrite('out.png', vis1)


