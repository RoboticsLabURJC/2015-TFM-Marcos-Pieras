import matplotlib.pyplot as plt
import numpy as np


grafOriginal = np.array([1,1,1,0.66,0.75,0.6,0.66,0.57,0.5,0.44,0.5])
recall = np.array([0,0.2,0.4,0.4,0.6,0.6,0.8,0.8,0.8,0.8,1])


grafOriginalaa = np.array([1,1,1,0.75,0.75,0.66,0.66,0.66,0.5,0.5,0.5])

#grafInterpolated = np.array([1,])

plt.plot(recall,grafOriginal,linewidth=5.0)
plt.plot(recall,grafOriginalaa,linewidth=5.0,linestyle='-', dashes=(5, 20))
plt.title('Precision Recall Curve',fontsize=50)
plt.ylabel('Precision',fontsize=40)
plt.xlabel('Recall',fontsize=40)
plt.ylim([0,1])
plt.xlim([0,1])
plt.tick_params(axis='y',labelsize=25)
plt.tick_params(axis='x',labelsize=25)
plt.legend(['Precision', 'Interpolated'], loc='upper right',fontsize=30)
plt.show()


