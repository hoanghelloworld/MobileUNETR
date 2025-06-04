import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, is_test=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.image_files.sort()
        
        if not is_test:
            self.mask_files = [f.replace('.jpg', '.png') for f in self.image_files]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            return {
                'image': image,
                'filename': self.image_files[idx]
            }
        else:
            # Load mask
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert mask to binary (0, 1)
            mask = (mask > 127).astype(np.uint8)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            return {
                'image': image,
                'mask': mask.float().unsqueeze(0)  # Add channel dimension
            }


def get_transforms(image_size=256, is_train=True):
    """Get data transforms"""
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
