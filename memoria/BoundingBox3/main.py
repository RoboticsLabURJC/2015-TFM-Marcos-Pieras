import cv2

img1 = cv2.imread('arrow.png',1)
img2 = cv2.imread('bounding.png',1)
img3 = cv2.imread('ppints.png',1)



framePintar1 = cv2.resize(img1,(403,564), interpolation = cv2.INTER_CUBIC)
framePintar2 = cv2.resize(img2,(403,564), interpolation = cv2.INTER_CUBIC)
framePintar3 = cv2.resize(img3,(403,564), interpolation = cv2.INTER_CUBIC)


cv2.imwrite('arrow1.png',framePintar1)
cv2.imwrite('bounding1.png',framePintar2)
cv2.imwrite('ppints1.png',framePintar3)