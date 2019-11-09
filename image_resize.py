import sys
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_pandas
import matplotlib.pyplot as plt

from utils import mask2rle, rle2mask

dataset_dir = './satellite_dataset/'
processed_dataset_dir = './satellite_dataset/preprocess_images'

'''
(height, width)
'''
original_size = (1400, 2100)
new_size = (384, 576)  # ResNet
interpolation = cv2.INTER_CUBIC

df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))

print('update train.csv ...')

# masking 영역을 ratio 맞게 resize 하기
for idx, row in tqdm(df.iterrows()):
    encoded_pixels = row[1]
    if encoded_pixels is not np.nan:
        mask = rle2mask(encoded_pixels, shape=original_size[::-1])
        mask = cv2.resize(mask, new_size[::-1], interpolation=interpolation)

        rle = mask2rle(mask)
        df.at[idx, 'EncodedPixels'] = rle

df.to_csv(os.path.join(dataset_dir, '384_576_train.csv'), index=False)


# Resizing Train and Test Images

train_images_dir = os.path.join(dataset_dir, 'train_images')
train_image_files = os.listdir(train_images_dir)

test_images_dir = os.path.join(dataset_dir, 'test_images')
test_image_files = os.listdir(test_images_dir)

preprocessed_train_image_dir = os.path.join(processed_dataset_dir, 'train_images')
preprocessed_test_image_dir = os.path.join(processed_dataset_dir, 'test_images')

print('resizing train images ...')

for image_file in tqdm(train_image_files):
    img = cv2.imread(os.path.join(train_images_dir, image_file))
    img = cv2.resize(img, new_size[::-1], interpolation=interpolation)
    dst = os.path.join(preprocessed_train_image_dir, image_file)
    cv2.imwrite(dst, img)


print('\n resizing test images ...')

for image_file in tqdm(test_image_files):
    img = cv2.imread(os.path.join(test_images_dir, image_file))
    img = cv2.resize(img, new_size[::-1], interpolation=interpolation)
    dst = os.path.join(preprocessed_test_image_dir, image_file)
    cv2.imwrite(dst, img)
