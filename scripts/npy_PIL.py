import matplotlib.pyplot as plt
import numpy as np

dfile = 'samples\samples_16x32x32x3.npz'
images = np.load(dfile)["arr_0"]
plt.ion()
plt.figure()
plt.imshow(images[0])