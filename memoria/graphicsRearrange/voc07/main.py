import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 20
means_frank = (600, 689, 950, 520,970,440,2500,710,1560,500,420,1000,700,670,9857,1000,500,486,650,680)
means_guido = (400, 480, 670, 350,450,350,1490,680,850,270,390,850,520,490,4500,490,190,490,500,490)

print(np.shape(means_frank))
print(np.shape(means_guido))
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, means_frank, bar_width,
                 alpha=opacity,
                 label='Object')
 
rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                 alpha=opacity,
                 label='Images')
 
#plt.xlabel('Person',fontsize=40)
#plt.ylabel('Scores',fontsize=40)
#plt.title('Distribution classes',fontsize=40)
plt.xticks(index + bar_width, ('plane', 'bike', 'bird', 'boat','bottle','bus','car','cat','chair','cow','table','dog','horse','bike','pers','plant','heep','sofa','train','tv'),fontsize=32)
plt.legend(fontsize=35)
plt.tick_params(axis='y',labelsize=25)
plt.tight_layout()
plt.yscale('log')
plt.show()