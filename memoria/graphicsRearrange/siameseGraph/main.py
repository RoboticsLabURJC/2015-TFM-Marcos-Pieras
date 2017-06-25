import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = plt.gca()

ax.scatter(0.0147,0.534,s=1980)
ax.scatter(0.01557,0.543,s=1980)
ax.scatter(0.0154,0.585,s=1980)
ax.scatter(0.0167,0.608,s=1980)
ax.scatter(0.032,0.5622,s=1980)

ax.scatter(0.0222,0.29,s=1980)

ax.scatter(0.731,0.43,s=1980)
ax.scatter(0.0222,0.506,s=1980)

ax.scatter(0.0331,0.414,s=1980)
ax.scatter(0.0345,0.52,s=1980)


#ax.set_xlim([0,1.0])
#ax.set_ylim([0,1.0])
#ax.set_yscale('log')
ax.set_xscale('log')
ax.tick_params(labelsize=25)
#ax.set_title('Number of categories vs. number of instances',fontsize=50)

ax.set_ylabel('rank1 score',fontsize=40)
ax.set_xlabel('time',fontsize=40)
ax.legend(['conv4','conv5','conv6','conv7','conv8','convSPP','inception_cosine','inception_tuned','siamese_cost','siamese_INnetwork'],loc='upper right',fontsize=30)
plt.show()

'''

fig = plt.figure()
ax = plt.gca()


ax.scatter(0.0147,0.34,s=1980)
ax.scatter(0.01557,0.37,s=1980)
ax.scatter(0.0158,0.46,s=1980)

ax.scatter(0.0167,0.45,s=1980)
ax.scatter(0.032,0.32,s=1980)
ax.scatter(0.026,0.35,s=1980)


ax.scatter(0.0222,0.42,s=1980)



ax.set_xlim([0,0.04])
#ax.set_ylim([0,1.0])
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.tick_params(labelsize=25)
#ax.set_title('Number of categories vs. number of instances',fontsize=50)

ax.set_ylabel('rank1 score',fontsize=40)
ax.set_xlabel('time',fontsize=40)
ax.legend(['conv_4','conv_5','conv_7','conv_8','cost_function','in_network','conv_SPP'],loc='upper right',fontsize=30)
plt.show()
'''