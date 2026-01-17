import os, io
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np


"""
Utilities to build per-dataset / per-task DataLoaders for polyp segmentation.

This module discovers subfolders under a root `dataroot` that contain
both `images/` and `masks/`, builds a `PolypDataset` for each such
subfolder, splits each dataset into train/val/test according to
`ratio_set`, and returns DataLoaders for each split.

Notes:
- `ImageFile.LOAD_TRUNCATED_IMAGES = True` is set to tolerate truncated
  image files encountered in some datasets.
- The function `get_loaders` returns `(task_num, dataset_loaders)` where
  `dataset_loaders` is a list of dicts with keys `name`, `order`, and
  the three DataLoaders.
"""


class PolypDataset(data.Dataset):
    """Dataloader for polyp segmentation tasks.

    Each instance loads paired image and mask files, applies resizing and
    normalization transforms, and returns `(image, mask)` tensors.
    """
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize  # training image size
        
        # read image file paths (only .jpg or .png)
        self.images = [image_root +'\\'+ f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        # read ground-truth mask file paths
        self.gts = [gt_root +'\\'+ f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        # sort paths to keep order consistent
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        
        # filter files to ensure each image has a matching mask
        self.filter_files()
        
        # dataset size
        self.size = len(self.images)
        
        # image transform: resize -> tensor -> ImageNet normalization
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        
        # ground-truth transform: resize -> tensor
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        # ensure same number of images and masks
        print(len(self.images), len(self.gts) )
        assert len(self.images) == len(self.gts)
        
        images = []
        gts = []
        
        # keep only pairs where image and mask have identical sizes
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        
        self.images = images
        self.gts = gts


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_image_mask_folder_paths(dataroot):
    """Find subfolders under `dataroot` that contain both `images` and `masks`.

    Returns a tuple (folder_paths, folder_names) where folder_paths is a
    list of `(image_folder, mask_folder)` pairs and folder_names contains
    the corresponding directory names.
    """
    folder_paths = []
    folder_names = []

    for root, dirs, files in os.walk(dataroot):
        if 'images' in dirs and 'masks' in dirs:
            image_folder = os.path.join(root, 'images')
            mask_folder = os.path.join(root, 'masks')
            folder_paths.append((image_folder, mask_folder))
            
            # folder name is used as dataset/task identifier
            folder_name = os.path.basename(root)
            folder_names.append(folder_name)

    return folder_paths, folder_names


def get_loaders(dataroot, batchsize, trainsize,ratio_set,num_workers=0, pin_memory=False, seed = 1):

    torch.manual_seed(seed)
    paths, names = get_image_mask_folder_paths(dataroot)
    total_dataset = []

    for (image_root, gt_root), name in zip(paths, names):
        dataset = PolypDataset(image_root, gt_root, trainsize)
        total_dataset.append((name, dataset))
            
    # number of discovered sub-datasets
    num_sets = len(paths)

    # split ratios
    train_ratio = ratio_set[0]
    val_ratio = ratio_set[1]
    test_ratio = ratio_set[2]

    dataset_loaders = []

    for name , dataset in total_dataset:
        # compute split sizes for this dataset
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        # split into train/val/test
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        
        # create DataLoaders; note train_loader currently uses shuffle=False
        train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=False, drop_last=True,num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_set, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        dataset_loaders.append({
            'name': name,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        })

    return dataset_loaders


if __name__ == '__main__':
    print("here")
    image_root = r'D:\PolypGen2021_MultiCenterData_v3'
 
    data_loaders = get_loaders(image_root, batchsize=8, ratio_set=[1.0,0.0,0.0],trainsize=256)
    print(data_loaders)

    for idx, loaders in enumerate(data_loaders):
            name =loaders['name']
            train_loader = loaders['train_loader']
            val_loader = loaders['val_loader']
            test_loader = loaders['test_loader']
            
            print(f"Dataset {name}:")
            print(f"  Training set: {len(train_loader.dataset)} samples in {len(train_loader)} batches")
            print(f"  Validation set: {len(val_loader.dataset)} samples in {len(val_loader)} batches")
            print(f"  Test set: {len(test_loader.dataset)} samples in {len(test_loader)} batches")