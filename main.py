from skimage import feature, io
import numpy as np
import matplotlib.pyplot as plt
import os

images_dir = 'Dataset/consumer'

filename = '!0ZEC4QB!F2DMB8DSA29BFM1JRQDG0.JPG'

image = io.imread(os.path.join(images_dir, filename))

plt.imshow(image)

plt.show()


