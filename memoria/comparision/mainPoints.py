import matplotlib.pyplot as plt
import numpy as np



'''
plt.scatter(7000, 20)
plt.scatter(40000,80)
plt.scatter(7000,200)
#plt.title('model accuracy')
plt.ylabel('Precision',fontsize=40)
plt.xlabel('Recall',fontsize=40)
plt.ylim([0,1])
plt.xlim([0,1])

plt.legend(['Precision', 'Interpolated'], loc='upper right',fontsize=30)
plt.show()
'''


fig = plt.figure()
ax = plt.gca()
ax.scatter(10.8,15.85,s=1980,label='Our')

ax.scatter(33.2,0.3,s=1980,label='CEM')
ax.scatter(32.2,212.6,s=1980,label='DP_NMS')
ax.scatter(29.7,0.2,s=1980,label='SMOT')
ax.scatter(26.2,22.2,s=1980,label='JPDA_M')


ax.scatter(48.8,0.5,s=1980,label='LMP')
ax.scatter(47.1,1.0,s=1980,label='MDPNN16')




#ax.set_xlim([1,1000000])
#ax.set_ylim([0.0,1.0])
ax.set_yscale('log')
#ax.set_xscale('log')
ax.tick_params(labelsize=30)
#ax.set_title('Number of categories vs. number of instances',fontsize=50)

ax.set_ylabel('Speed',fontsize=40)
ax.set_xlabel('MOTA',fontsize=40)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=40)
#ax.legend(['SSD_MOBILENET','SSD_INCEPTION','SSD_VGG','RFCN_RESNET','FasterRCNN_RESNET','FasterRCNN_ENSEMBLE'],loc='upper right',fontsize=30)
plt.show()