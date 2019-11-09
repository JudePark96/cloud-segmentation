import os

processed_dir = './satellite_dataset/preprocess_images'
dataset_dir = './satellite_dataset'

assert len(os.listdir(os.path.join(dataset_dir, 'train_images'))) == len(os.listdir(os.path.join(processed_dir, 'train_images')))
assert len(os.listdir(os.path.join(dataset_dir, 'test_images'))) == len(os.listdir(os.path.join(processed_dir, 'test_images')))