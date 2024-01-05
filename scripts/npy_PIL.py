import matplotlib.pyplot as plt
import numpy as np

dfile = 'samples/result.npz'
images = np.load(dfile)["arr_0"]
plt.ion()
plt.figure()
plt.imshow(images[0])
import pdb; pdb.set_trace()