import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt


sats =  np.loadtxt('points.txt',delimiter = ',')

sizOO = 298
siz =  range(0,sizOO)
print(np.shape(sats),np.shape(sats[1:299,0]),np.shape(sats[1:299,1]),np.shape(siz))
'''
plt.bar(siz,sats[1:299,0], color='y')
plt.plot(sats[1:299,1])
plt.show()
'''
'''
fig = plt.figure()

plt.bar(siz,sats[:,0]+sats[:,1]+sats[:,2]+sats[:,3], color='r')
plt.bar(siz,sats[:,0]+sats[:,1]+sats[:,2], color='g')
plt.bar(siz,sats[:,0]+sats[:,1], color='b')
plt.bar(siz,sats[:,0], color='y')
plt.legend(('Save Data', 'Tracking','Data association','Read frame'),fontsize=30,loc='upper right')
plt.xlabel('Frames',fontsize=30)
plt.ylabel('Time',fontsize=30)
plt.tick_params(labelsize=20)
plt.plot(vale,'r')
#plt.ylim([0,0.2])
plt.show()

print(np.mean(sats[:,0]+sats[:,1]+sats[:,2]+sats[:,3]))
'''


import numpy as np
import matplotlib.pyplot as plt


def two_scales(ax1, time, data1, data2, c1, c2):
    """

    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    time : array-like
        x-axis values for both datasets

    data1: array-like
        Data for left hand scale

    data2 : array-like
        Data for right hand scale

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('Frames',fontsize=30)
    ax1.set_ylabel('Number of Points',fontsize=30)
    ax1.tick_params(labelsize=20)

    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel('Time',fontsize=30)
    ax2.tick_params(labelsize=20)
    return ax1, ax2


# Create some mock data
t = range(0,sizOO)
s1 = sats[1:299,0]
s2 = sats[1:299,1]

# Create axes
fig, ax = plt.subplots()
ax1, ax2 = two_scales(ax, t, s1, s2, 'r', 'b')


# Change color of each axis
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None
color_y_axis(ax1, 'r')
color_y_axis(ax2, 'b')
plt.tick_params(labelsize=20)
plt.show()