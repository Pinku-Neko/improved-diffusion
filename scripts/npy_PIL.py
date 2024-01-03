import matplotlib.pyplot as plt
import numpy as np

dfile = 'samples\samples_16x32x32x3_1.npz'
images = np.load(dfile)["arr_0"]
plt.ion()
plt.figure()
plt.imshow(images[0])
import pdb; pdb.set_trace()