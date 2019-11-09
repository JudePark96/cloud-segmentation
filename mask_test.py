import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = np.array(cv2.imread('./satellite_dataset/preprocess_images/gt_train_images/Fish_0a7a247.jpg'))
print(data.shape)

plt.imshow(data)
plt.show()

