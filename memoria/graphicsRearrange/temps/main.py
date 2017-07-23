import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

'''
sats =  np.loadtxt('temps3.txt',delimiter = ',')
print(np.shape(sats),np.shape(sats[:,0]),np.shape(sats[:,1]),np.shape(sats[:,2]),np.shape(sats[:,3]))
sizOO = 29
siz =  range(0,sizOO)
vale = np.ones(sizOO)
vale = vale*0.064287

fig = plt.figure()

plt.bar(siz,sats[:30,0]+sats[:,1]+sats[:,2]+sats[:,3], color='r')
plt.bar(siz,sats[:,0]+sats[:,1]+sats[:,2], color='g')
plt.bar(siz,sats[:,0]+sats[:,1], color='b')
plt.bar(siz,sats[:,0], color='y')
plt.legend(('Save Data', 'Feature-based tracking','Data association','Read frame'),fontsize=30,loc='upper right')
plt.xlabel('Frames',fontsize=30)
plt.ylabel('Time',fontsize=30)
plt.tick_params(labelsize=20)
#plt.plot(vale,'r')
plt.ylim([0,0.2])
plt.show()

print(np.mean(sats[:,0]+sats[:,1]+sats[:,2]+sats[:,3]))
'''
sats =  np.loadtxt('temps3.txt',delimiter = ',')
print(np.shape(sats),np.shape(sats[:30,0]),np.shape(sats[:30,1]),np.shape(sats[:30,2]),np.shape(sats[:30,3]))
sizOO = 29
siz =  range(0,sizOO)
vale = np.ones(sizOO)
vale = vale*0.064287

fig = plt.figure()

T1 = 270
T2 = 299
print(np.shape(sats[T1:T2,0]))
print(np.mean(sats[T1:T2,0]+sats[T1:T2,1]+sats[T1:T2,2]+sats[T1:T2,3]))

plt.bar(siz,sats[T1:T2,0]+sats[T1:T2,1]+sats[T1:T2,2]+sats[T1:T2,3], color='r')
plt.bar(siz,sats[T1:T2,0]+sats[T1:T2,1]+sats[T1:T2,2], color='g')
plt.bar(siz,sats[T1:T2,0]+sats[T1:T2,1], color='b')
plt.bar(siz,sats[T1:T2,0], color='y')
plt.legend(('Save Data', 'Feature-based tracking','Data association','Read frame'),fontsize=30,loc='upper right')
plt.xlabel('Frames',fontsize=30)
plt.ylabel('Time',fontsize=30)
plt.tick_params(labelsize=20)
#plt.plot(vale,'r')
plt.ylim([0,0.2])
plt.show()

print(np.mean(sats[:,0]+sats[:,1]+sats[:,2]+sats[:,3]))