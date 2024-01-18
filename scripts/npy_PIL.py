import matplotlib.pyplot as plt
import numpy as np

dfile = 'samples/samples_20240105_151856_skip0.8_5000.npz'
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