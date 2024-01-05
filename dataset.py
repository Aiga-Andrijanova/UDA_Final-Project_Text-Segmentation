import os
import random
import re
import cv2 
import torch

import numpy as np

class Sample():
    """
    A class that represents a single sample.
    """
    def __init__(self, img, mask):
        self.img = img
        self.mask = mask
        
        ids_list = re.findall(r'\d+', img)
        if len(ids_list) >= 3:
            self.name = ids_list[-3] + "_" + ids_list[-2] + "_" + ids_list[-1]
        else:
            print(f"Unable to set self.name for img with path: {img}")


def dataset_splitter(data_paths: list, train_split_size: float=0.7, percentage_of_data_to_use:float=1, seed=0): 
    """
    This function splits the dataset into training and validation sets based on document ids.

    Args:
        data_paths: a list of root folders of the generated and tiled samples
        train_split_size: ratio of how much of the documents put in the train set
        percentage_of_data_to_use: how much of the total samples to use, intended for debugging purposes when only a small dataset is needed
        seed: random seed

    Returns:
        train_items: a list of training items of class Sample
        val_items: a list of validation items of class Sample
    """
    train_items = []
    val_items = []

    random.seed(seed)

    for data_path in data_paths:
        subdirs = [x[0] for x in os.walk(data_path)]
        del subdirs[0]
        random.shuffle(subdirs)
        subdirs = subdirs[:int(len(subdirs)*percentage_of_data_to_use)]
        train_len = int(len(subdirs) * train_split_size)
        
        pattern = re.compile("image_\d+_\d+.png")
        train_subdirs = 0
        for subdir in subdirs:
            if train_subdirs < train_len:
                items_list = train_items
                train_subdirs += 1
            else:
                items_list = val_items
            for file in os.listdir(subdir):
                if pattern.match(file):
                        items_list.append(Sample(os.path.join(subdir, file), os.path.join(subdir, file.replace("image", "mask"))))
    return train_items, val_items


class PaperworkDataset(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that represents a single Paperwork dataset.
    """
    def __init__(self, training_items, input_shape, transformations, tile_size=512):
        self.input_shape = input_shape
        self.input_items = training_items
        self.tile_size = tile_size

        self.transformations = transformations
        
    def __len__(self):
        return len(self.input_items)
    
    def read_image(self, img_path):
        img = cv2.imread(img_path)
        orig_img_size = img.shape
        return img, orig_img_size

    def read_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = np.clip(mask, .0, 1.)
        return mask

    def __getitem__(self, index):
        img, _ = self.read_image(self.input_items[index].img)
        mask = self.read_mask(self.input_items[index].mask)
        name = self.input_items[index].name

        aug = self.transformations(image=img, mask=mask)
        img = aug['image']
        mask = aug['mask']

        img = img / 255.0
        return img, mask, name
