import os
import sys

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

TRAIN_DIR = os.path.join(os.path.dirname(__file__), 'train')
VAL_DIR = os.path.join(os.path.dirname(__file__), 'val')

TRAIN_BATCHES = [f'train_data_batch_{i}' for i in range(1, 11)]
VAL_BATCH = 'val_data'

def unpickle(file): 
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def process_loaded_batch(data, labels): 
    images = data.reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
    labels = np.array(labels) - 1
    return images, labels

def process_batch_file(file): 
    data = unpickle(file)
    images, labels = process_loaded_batch(data['data'], data['labels'])
    image_label_pairs = list(zip(images, labels))
    return image_label_pairs

def train_proc(): 
    count_each_label = defaultdict(int)
    for batch in TRAIN_BATCHES: 
        file_path = os.path.join(TRAIN_DIR, batch)
        image_label_pairs = process_batch_file(file_path)
        for image, label in tqdm(image_label_pairs, desc=f'Processing {batch} data'):
            if not os.path.exists(f'{TRAIN_DIR}/{label}'):
                os.makedirs(f'{TRAIN_DIR}/{label}')
            plt.imsave(f'{TRAIN_DIR}/{label}/{count_each_label[label]}.png', image)
            count_each_label[label] += 1
        os.remove(file_path)

def val_proc(): 
    file_path = os.path.join(VAL_DIR, VAL_BATCH)
    image_label_pairs = process_batch_file(file_path)
    count_each_label = defaultdict(int)
    for image, label in tqdm(image_label_pairs, desc='Processing validation data'):
        if not os.path.exists(f'{VAL_DIR}/{label}'):
            os.makedirs(f'{VAL_DIR}/{label}')
        plt.imsave(f'{VAL_DIR}/{label}/{count_each_label[label]}.png', image)
        count_each_label[label] += 1
    os.remove(file_path)

if __name__ == '__main__': 
    train_proc()
    val_proc()
