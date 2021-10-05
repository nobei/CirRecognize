import time

import h5py
import scipy.io as scio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mat4py
import tables



timeStart = time.time()

path = "E:/java/saveDataFromClient/src/audio/upDown/1624799323514upDown.mat"


data = tables.open_file(path)


timeEnd = time.time()

print(timeEnd-timeStart)

path1 = "E:/java/saveDataFromClient/src/audio/double click/1622619878992double click"

data = scio.loadmat(path)

timeNew = time.time()

print(timeNew-timeEnd)

path1 = "E:/java/saveDataFromClient/src/audio/double click/1622605210302double click.mat"

# f = tables.open_file(path)

timeNew2 = time.time()





data = np.array(data['x'])



plt.plot(data)
plt.show()

sns.heatmap(data, cmap='Reds')
plt.show()