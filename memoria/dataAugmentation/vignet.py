import cv2
import numpy as np

img = cv2.imread('roifram1.png')
imageOriginal = cv2.resize(img,(64, 128), interpolation = cv2.INTER_CUBIC)
rows, cols = imageOriginal.shape[:2]

# generating vignette mask using Gaussian kernels
#kernel_x = cv2.getGaussianKernel(cols,30)
#kernel_y = cv2.getGaussianKernel(rows,60)

kernel_x = cv2.getGaussianKernel(cols,30)
kernel_y = cv2.getGaussianKernel(rows,40)

kernel = kernel_y * kernel_x.T
mask = 1/(255 * kernel / np.linalg.norm(kernel))
output = np.copy(imageOriginal)



output[:,:,0] = output[:,:,0] * mask
output[:,:,1] = output[:,:,1] * mask
output[:,:,2] = output[:,:,2] * mask
output.astype(np.uint8)

maxNUM = np.amax(output)
minNUM = np.amin(output)
#print(maxNUM,minNUM)
#noisy2 = (255*(output-minNUM)/(maxNUM-minNUM))

cv2.imwrite('imageBLURcenter.jpg',output)