




import cv2









img1b = cv2.imread('/home/marc/Dropbox/tfmDeepLearning/Presentació/aplicaciones/policias.jpg')
img2b = cv2.imread('/home/marc/Dropbox/tfmDeepLearning/Presentació/aplicaciones/Seleccio_007.png')
img3b = cv2.imread('/home/marc/Dropbox/tfmDeepLearning/Presentació/aplicaciones/singing.jpg')
img4b = cv2.imread('/home/marc/Dropbox/tfmDeepLearning/Presentació/aplicaciones/tomeu.png')



img1 = cv2.resize(img1b,(640,320), interpolation = cv2.INTER_CUBIC)
img2 = cv2.resize(img2b,(640,320), interpolation = cv2.INTER_CUBIC)
img3 = cv2.resize(img3b,(640,320), interpolation = cv2.INTER_CUBIC)
img4 = cv2.resize(img4b,(640,320), interpolation = cv2.INTER_CUBIC)


cv2.imwrite('policias2.png', img1)
cv2.imwrite('selecio2.png', img2)
cv2.imwrite('singing2.png', img3)
cv2.imwrite('tomeu2.png', img4)
