import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import imutils

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def randomCrop(image):

	f = np.random.randint(0,2,1)
	f2 = np.random.randint(0,2,1)
	if f == 1:
	#vertical crop
		middlePoint = int(np.shape(image)[0]/2.0)
		
		if f2 == 1:
		#left
			initial = 0
			point = int(initial+middlePoint*np.random.uniform())

			image[initial:point,:,:]=0.0
		else:

			initial = middlePoint
			point = int(initial+int(2*middlePoint*np.random.uniform()))

			image[int(initial):int(point),:,:]=0.0

	else:
	# horizontal crop
		middlePoint = int(np.shape(image)[1]/2.0)
		
		if f2 == 1:
		#left
			initial = 0
			point = int(initial+middlePoint*np.random.uniform())

			image[:,initial:point,:]=0.0
		else:

			initial = middlePoint
			point = int(initial+int(2*middlePoint*np.random.uniform()))

			image[:,initial:point,:]=0.0

	return image

def addGaussianNoise(image):
    row,col,ch= image.shape
    mean = 128
    gauss = np.random.normal(mean,15,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    
    noisy = image + gauss
    maxNUM = np.amax(noisy)
    minNUM = np.amin(noisy)
    noisy2 = (255*(noisy-minNUM)/(maxNUM-minNUM))
    return noisy2

def verticalFlip(image):
	rimg=cv2.flip(image,1)
	return rimg

def gaussianBlur(image):
	blur = cv2.GaussianBlur(image,(7,7),0)
	return blur

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1

    if np.random.randint(2)==1:
    	random_bright = .5
    	cond1 = shadow_mask==1
    	cond0 = shadow_mask==0
    	if np.random.randint(2)==1:
    		image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
    	else:
    		image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright

    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def zoomIn(cv2Object, zoomSize):
    # Resizes the image/video frame to the specified amount of "zoomSize".
    # A zoomSize of "2", for example, will double the canvas size
    col,row,ch = np.shape(cv2Object)
    cv2Object = imutils.resize(cv2Object, width=(int(zoomSize * cv2Object.shape[1])))
    # center is simply half of the height & width (y/2,x/2)
    center = (cv2Object.shape[0]/2,cv2Object.shape[1]/2)
    # cropScale represents the top left corner of the cropped frame (y/x)
    cropScale = (center[0]/zoomSize, center[1]/zoomSize)
    # The image/video frame is cropped to the center with a size of the original picture
    # image[y1:y2,x1:x2] is used to iterate and grab a portion of an image
    # (y1,x1) is the top left corner and (y2,x1) is the bottom right corner of new cropped frame.
    cv2Object = cv2Object[int(cropScale[0]):(int(center[0]) + int(cropScale[0])), int(cropScale[1]):(int(center[1]) + int(cropScale[1]))]
    imageOriginal = cv2.resize(cv2Object,(row, col), interpolation = cv2.INTER_CUBIC)
    return imageOriginal

def zoomIn2(cv2Object, zoomSize):
    
    col,row,ch = np.shape(cv2Object)
    cv2Object = cv2.resize(cv2Object,(int(col*zoomSize), int(row*zoomSize)), interpolation = cv2.INTER_CUBIC)
    
    center = (cv2Object.shape[0]/2,cv2Object.shape[1]/2)
    cropScale = (center[0]/zoomSize, center[1]/zoomSize)
    cv2Object = cv2Object[int(cropScale[0]):(int(center[0]) + int(cropScale[0])), int(cropScale[1]):(int(center[1]) + int(cropScale[1]))]
    imageOriginal = cv2.resize(cv2Object,(row, col), interpolation = cv2.INTER_CUBIC)

    return imageOriginal

def zoomOut(cv2Object):
    imageOriginal = cv2.resize(cv2Object,(32, 64), interpolation = cv2.INTER_CUBIC)
    constant= cv2.copyMakeBorder(imageOriginal,10,10,10,10,cv2.BORDER_CONSTANT,value=(0,0,0))
    cv2ObjectNEw = cv2.resize(constant,(64, 128), interpolation = cv2.INTER_CUBIC)

    return cv2ObjectNEw


def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
    # Brightness 
    

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    img = augment_brightness_camera_images(img)
    
    return img

def vignetInv(image):
    rows, cols = image.shape[:2]

    # generating vignette mask using Gaussian kernels
    #kernel_x = cv2.getGaussianKernel(cols,30)
    #kernel_y = cv2.getGaussianKernel(rows,60)

    kernel_x = cv2.getGaussianKernel(cols,30)
    kernel_y = cv2.getGaussianKernel(rows,40)

    kernel = kernel_y * kernel_x.T
    mask = 1/(255 * kernel / np.linalg.norm(kernel))
    output = np.copy(image)



    output[:,:,0] = output[:,:,0] * mask
    output[:,:,1] = output[:,:,1] * mask
    output[:,:,2] = output[:,:,2] * mask
    output.astype(np.uint8)
    return output

image = cv2.imread('roifram1.png')
#imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imageOriginal = cv2.resize(image,(64, 128), interpolation = cv2.INTER_CUBIC)
#cv2.imwrite('resizedImage.jpg',imageOriginal)

'''
imageBrightness = augment_brightness_camera_images(imageOriginal)
cv2.imwrite('imageBrightnes.jpg',imageBrightness)



imageRandom = randomCrop(imageOriginal)
cv2.imwrite('imageRandomCrop.jpg',imageRandom)



imageVerticalFlip = verticalFlip(imageOriginal)
cv2.imwrite('imageVerticalFlip.jpg',imageVerticalFlip)



imageBlur = gaussianBlur(imageOriginal)
cv2.imwrite('imageGaussianBlur.jpg',imageBlur)
imageRandomShadow = add_random_shadow(imageOriginal)
cv2.imwrite('imageRandomShadow.jpg',imageRandomShadow)

imageZoomIn = zoomIn(imageOriginal, 5)
cv2.imwrite('imageZoomIn.jpg',imageZoomIn)

imageTransformed = transform_image(imageOriginal,20,10,5)
cv2.imwrite('imageTransormed.jpg',imageTransformed)

zoomOut = zoomOut(imageOriginal,2)
cv2.imwrite('imageZoomOut.jpg',zoomOut)

gaussianNoise  = addGaussianNoise(imageOriginal)
cv2.imwrite('imageNoiseGaussian.jpg',gaussianNoise)
'''

#imageVig = vignetInv(imageOriginal)
#cv2.imwrite('imageBLURcen2ter.jpg',imageVig)


imageZoomIn = zoomIn2(imageOriginal, 5)
cv2.imwrite('imageZoomIn222.jpg',imageZoomIn)
