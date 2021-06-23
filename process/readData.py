
import scipy.io as scio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

path = "E:/java/saveDataFromClient/src/audio/double click/1622602220097double click"

data = scio.loadmat(path)

data = np.array(data['x'])

plt.plot(data)
plt.show()

sns.heatmap(data, cmap='Reds')
plt.show()