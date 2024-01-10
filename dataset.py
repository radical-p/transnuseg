import scipy.io as sio
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob
import cv2
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



class MyDataset(Dataset):
    '''
    dir_path: path to data, having two folders named data and label respectively
    '''
    def __init__(self,dir_path,transform = None,in_chan = 3): 
        self.dir_path = dir_path
        self.transform = transform
        self.data_path = os.path.join(dir_path,"data")
        self.data_lists = sorted(glob.glob(os.path.join(self.data_path,"*.png")))
        self.label_path = os.path.join(dir_path,"label")
        self.label_lists = sorted(glob.glob(os.path.join(self.label_path,"*.png")))
        
        self.in_chan = in_chan
        
    def __getitem__(self, index):
        img_path = self.data_lists[index]
        label_path = self.label_lists[index]

        # Load images
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if self.in_chan == 3 else cv2.COLOR_BGR2GRAY)

        # Load and process label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # Process masks
        semantic_mask = np.where(label == 255, 1, 0)
        instance_mask = self.sem2ins(semantic_mask.copy())
        normal_edge_mask = self.generate_normal_edge_mask(label)
        cluster_edge_mask = self.generate_cluster_edge_mask(label)

        img = img / 255.0 if np.max(img) > 2 else img

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=img, masks=[instance_mask, semantic_mask, normal_edge_mask, cluster_edge_mask])
            img = augmented['image']
            instance_mask, semantic_mask, normal_edge_mask, cluster_edge_mask = augmented['masks']

        # Convert to tensor

        #instance_mask = instance_mask / 255.0 if np.max(instance_mask) > 2 else instance_mask


        img = ToTensorV2()(image=img)['image']
        instance_mask = torch.tensor(instance_mask, dtype=torch.float32)
        semantic_mask = torch.tensor(semantic_mask, dtype=torch.float32)
        normal_edge_mask = torch.tensor(normal_edge_mask, dtype=torch.float32)
        cluster_edge_mask = torch.tensor(cluster_edge_mask, dtype=torch.float32)

        # Send to device
        #img = img / 255.0 if np.max(img) > 2 else img
        img = img.to(device)

        #instance_mask = instance_mask / 255.0 if np.max(instance_mask) > 2 else instance_mask
        instance_mask = instance_mask.to(device)

        #semantic_mask = semantic_mask / 255.0 if np.max(semantic_mask) > 2 else semantic_mask
        semantic_mask = semantic_mask.to(device)

        #normal_edge_mask = normal_edge_mask / 255.0 if np.max(normal_edge_mask) > 2 else normal_edge_mask
        normal_edge_mask = normal_edge_mask.to(device)

        #cluster_edge_mask = cluster_edge_mask / 255.0 if np.max(cluster_edge_mask) > 2 else cluster_edge_mask
        cluster_edge_mask = cluster_edge_mask.to(device)

        return img, instance_mask, semantic_mask, normal_edge_mask, cluster_edge_mask
    
    def __len__(self):
        return len(self.data_lists)

    
    def sem2ins(self,label):
        seg_mask_g = label.copy()

        # Ensure the mask is in the correct format (8-bit single-channel)
        seg_mask_g = np.uint8(seg_mask_g * 255)  # Assuming your mask is in [0, 1] range

        contours, hierarchy = cv2.findContours(seg_mask_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            cnt = contours[i]
            cv2.drawContours(seg_mask_g, [cnt], 0, i+1, -1)

        return seg_mask_g
    
    def generate_normal_edge_mask(self,label):
        
        normal_edge_mask = label.copy()

        normal_edge_mask[normal_edge_mask == 150] = 2
        normal_edge_mask[normal_edge_mask == 76] = 2
        normal_edge_mask[normal_edge_mask != 2] = 0
        normal_edge_mask[normal_edge_mask == 2] = 1

        

        return normal_edge_mask
    def generate_cluster_edge_mask(self,label):
        
        cluster_edge_mask = label.copy()

        cluster_edge_mask[cluster_edge_mask != 76] = 0
        cluster_edge_mask[cluster_edge_mask == 76] = 1

        

        return cluster_edge_mask

