import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import mask2rle, rle2mask

import matplotlib.pyplot as plt

original_size = (1400, 2100)
new_size = (350, 525)  # ResNet

dataset_dir = './satellite_dataset/'
processed_dataset_dir = './satellite_dataset/preprocess_images'

df = pd.read_csv(os.path.join(dataset_dir, '350_525_train.csv'))

for idx, row in tqdm(df.iterrows()):
    encoded_pixels = row[1]
    names = row[0].split('_')   # i.g) 021401240.jpg, Flower
    filename = names[1] + '_' + names[0]

    if encoded_pixels is not np.nan:
        mask = rle2mask(encoded_pixels, new_size[::-1])
        plt.imsave(os.path.join(processed_dataset_dir + '/gt_train_images', filename), mask, cmap='gray')
