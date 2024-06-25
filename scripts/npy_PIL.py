import matplotlib.pyplot as plt
import numpy as np

dfile = 'samples/0315_ddim/samples_16x32x32x3.npz'
breakpoint()
images = np.load(dfile)["arr_0"]
plt.ion()
plt.figure()
try:
    for i in range(len(images)):
        plt.imshow(images[i])
        plt.pause(1)
        plt.draw()
except KeyboardInterrupt:
    print("Execution manually stopped by the user.")