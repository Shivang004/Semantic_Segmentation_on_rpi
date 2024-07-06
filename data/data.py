import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter

class ADE20kSegmentationDataset(Dataset):
    def __init__(self, root='/kaggle/input/ade20k-scene-parsing-extracted', split='training', transform=None,
                 base_size=520, crop_size=480):
        self.root = root
        self.split = split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size

        self.images_folder = os.path.join(self.root, 'images', self.split)
        self.annotations_folder = os.path.join(self.root, 'annotations', self.split)
        self.valid_classes = [1, 4, 6, 8, 9, 11, 13, 15, 16, 20, 24, 25, 31, 34, 36, 50, 54, 63, 70, 74]
        self.class_map = {class_id: index + 1 for index, class_id in enumerate(self.valid_classes)}
        self._load_images_and_masks()

    def _load_images_and_masks(self):
        self.image_paths = [os.path.join(self.images_folder, fname) for fname in os.listdir(self.images_folder) if fname.endswith('.jpg')]
        self.mask_paths = [os.path.join(self.annotations_folder, fname) for fname in os.listdir(self.annotations_folder) if fname.endswith('.png')]

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])

        if self.split == 'training':
            img, mask = self._sync_transform(img, mask)
        elif self.split == 'validation':
            img, mask = self._val_sync_transform(img, mask)

        if self.transform:
            img = self.transform(img)

        return img, mask

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        x1 = random.randint(0, img.width - self.crop_size)
        y1 = random.randint(0, img.height - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        x1 = int(round((img.width - outsize) / 2.))
        y1 = int(round((img.height - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        mask_np = np.array(mask)
        mask_np_mapped = np.zeros_like(mask_np, dtype=np.int32)  # Initialize with zeros

        for class_id, mapped_index in self.class_map.items():
            mask_np_mapped[mask_np == class_id] = mapped_index

        return torch.LongTensor(mask_np_mapped)

    def __len__(self):
        return len(self.image_paths)

