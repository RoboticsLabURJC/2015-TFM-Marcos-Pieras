import numpy as np
import cv2


anteriroLK = [[495 ,477, 589,728],[575,475, 695, 760],
 [1377 ,428 ,1530 ,794],
 [206 ,260, 306, 483],
 [986 ,5, 1070, 161],
 [354 ,158, 412, 301],
 [730 ,143, 792, 264],
 [189, 221, 247, 391]]

anteriroLK = np.array(anteriroLK)
numAnterior = np.shape(anteriroLK)
print(numAnterior)
detection = [[1631, 585, 1746, 860],
 [1085 ,1 ,1138 ,98],
 [626, 444 ,730 ,697],
 [867, 347 ,950 ,547],
 [171, 494 ,293 ,732],
 [1556, 91 ,1622 ,218],
 [519, 430 ,615 ,676],
 [806, 132 ,867 ,281],
 [1625, 412 ,1688 ,543]]


detection = np.array(detection)
numDetection = np.shape(detection)
print(numDetection)
imageAnterior = cv2.imread('frame414.jpg')
imageDetection = cv2.imread('frame422.jpg')


for i in range(0,numAnterior[0]):
	cv2.rectangle(imageDetection,(int(anteriroLK[i,0]),int(anteriroLK[i,1])),(int(anteriroLK[i,2]),int(anteriroLK[i,3])),(255,0,0),2)

for i in range(0,numDetection[0]):
	cv2.rectangle(imageDetection,(int(detection[i,0]),int(detection[i,1])),(int(detection[i,2]),int(detection[i,3])),(0,255,0),2)



while True:
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image',imageAnterior)
	cv2.resizeWindow('image', 1200,800)
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

while True:
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image',imageDetection)
	cv2.resizeWindow('image', 1200,800)
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break