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
ax.scatter(20,7000,s=1980)
ax.scatter(80,40000,s=1980)
ax.scatter(200,7000,s=1980)
ax.set_xlim([1,1000000])
ax.set_ylim([1,100000])
ax.set_yscale('log')
ax.set_xscale('log')
ax.tick_params(labelsize=25)
#ax.set_title('Number of categories vs. number of instances',fontsize=50)

ax.set_ylabel('Instances per category',fontsize=40)
ax.set_xlabel('Number of categories',fontsize=40)
ax.legend(['VOC','COCO','ImageNet'],loc='upper right',fontsize=30)
plt.show()