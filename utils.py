#!/usr/bin/env python

# import relevant packages
from functools import lru_cache
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
# from torchvision import transforms
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from albumentations import Resize
from albumentations.pytorch import ToTensorV2
import albumentations
import cv2
import random
import json
# import pydicom as dicom

RANZCR_CLIP_PATH = "/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/RANZCR_CLIP/"
MIMIC_CXR_PATH = "/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/"
RETINAL_PATH = "/home/ubuntu/dp_snow_0_3/"

class RETINAL(Dataset):
    """
    Brazilian Retinal Image Dataset Class
    """
    def __init__(self, df_all, retinal_path, transform=None):
     #   df_subset = df_all[df_all["image_id"].isin(df_studyIDs[0])]
        df_subset = df_all
        self.studyuid = df_subset["image_id"].values
        self.labels = df_subset['Class'].values
        self.transform = transform
        self.retinal_path = retinal_path
        
    def __len__(self):
        return self.studyuid.shape[0]
    
    def __getitem__(self, idx):
        path = self.studyuid[idx]
        
        # path = RETINAL_PATH + path+ ".jpg"
        path = os.path.join(self.retinal_path, path+'.jpg')
      #  print(path)
        image = cv2.imread(path)
       # print(image)
        image = self.transform(image=image)['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensorV2()(image = image)["image"]
        labels = self.labels[idx]
        return image, labels



# define datasets
class RANZCR_CLIP(Dataset):
    """RANZCR-CLIP Dataset Class

        Parameters
        ----------
        df_all : pandas DataFrame
            Contains all available labeled data (original train.csv format)
        df_studyIDs : pandas DataFrame
            lists all StudyInstanceUIDs to be used for this subset of RANZCR-CLIP
                (i.e. train IDs or val IDs)
        transform : transform types (i.e. what get_transform() returns)
    """
    def __init__(self, df_all, df_studyIDs, transform=None):
        df_subset = df_all[df_all["StudyInstanceUID"].isin(df_studyIDs[0])]
        self.studyuid = df_subset["StudyInstanceUID"].values
        self.labels = df_subset[LABELS].values
        self.transform = transform
        
    def __len__(self):
        return self.studyuid.shape[0]
    
    def __getitem__(self, idx):
        path = self.studyuid[idx]
        path = RANZCR_CLIP_PATH + "train/" + path + ".jpg"
        image = cv2.imread(path)
        image = self.transform(image=image)['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensorV2()(image = image)["image"]
        labels = self.labels[idx]
        return image, labels

class MIMIC_CXR(Dataset):
    '''MIMIC-CXR Dataset Class

        Parameters
        ----------
        df : pandas DataFrame
            Contains all available labeled data (similar to the original train.csv format of RANZCR-CLIP)
            Except that the first column should be the jpg path of the image
        transform : transform types (i.e. what get_transform() returns)
    '''
    def __init__(self, df, transform=None):
        self.labels = df[LABELS].values
        self.transform = transform
        self.path = df["StudyInstanceUID"].values

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        img_path = os.path.join(MIMIC_CXR_PATH, self.path[idx]).replace('dcm', 'jpg')
        image = cv2.imread(img_path)
        image = self.transform(image=image)['image']
        image = ToTensorV2()(image = image)["image"]
        labels = self.labels[idx]
        return image, labels

# define seeding function
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Setting all seeds to be {seed} to reproduce...')

# define visualization functions

# define logging functions

# define misc functions
LABELS = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present'
    ]

def readData(path):
    """Reads RANZCR-CLIP Train and Val data into DataFrames

    Parameters
    ----------
    path : str
        The path to the RANZCR-CLIP dataset

    Returns
    -------
    df_all pandas.DataFrame
        Contains all available labeled data (original RANZCR-CLIP train.csv format)
    df_train_val pandas.DataFrame
        Contains ALL StudyInstanceUIDs for both train and validation

    """
    print("Reading Data...")
    # df_all = pd.read_csv(os.path.join(path, "labelssubset.csv"))
    # df_train_val = pd.read_csv('train.txt', header=None)
    # print("Done!")
    # return df_all, df_train_val
    # df_all = pd.read_csv('/home/ubuntu/df1_train.csv')
    df_all = pd.read_csv(path)
    return df_all
def readTestData(path):
    """Reads RANZCR-CLIP Train and Val data into DataFrames
    Parameters
    ----------
    path : str
        The path to the RANZCR-CLIP dataset
    Returns
    -------
    df_all pandas.DataFrame
        Contains all available labeled data (original RANZCR-CLIP train.csv format)
    df_train_val pandas.DataFrame
        Contains ALL StudyInstanceUIDs for both train and validation
    """
    print("Reading Data...")
    # df_all = pd.read_csv(os.path.join(path, "labelssubset.csv"))
    # df_train_val = pd.read_csv('train.txt', header=None)
    # print("Done!")
    # return df_all, df_train_val
    # df_all = pd.read_csv('/home/ubuntu/df1_test.csv')
    df_all = pd.read_csv(path)
    return df_all

def k_fold_cross_val(df_train_val, df_all, k=3, stratified_grouped=False, val_perc=None):
    """Creates folds for cross validation or one split of specified percentage

    Parameters
    ----------
    df_train_val : pandas.DataFrame
        Contains ALL StudyInstanceUIDs for both train and validation
    k : int
        number of folds for cross validation (default is 3)
    stratified : boolean
        whether to preserve class distributions (default is False)
    grouped : boolean
        whether to control for patientIDs (no repeats across train and val sets)
            (default is False)
    val_perc : float
        fraction of total train_val set to use for validation (default is None)
            Only if you want to force a single iteration

    Returns
    -------
    folds : list
        list of length k with train and val StudyInstanceUIDs [[train1, val1], [train2, val2], ... ]
            Note: for non-NULL val_perc, folds is of length 1

    """
    folds = []
    if val_perc:
        train, val = train_test_split(df_train_val, test_size = val_perc, random_state=871)
        return [train, val]
    elif stratified_grouped:
        # encode each unique set of outcomes as a unique int
        enc = LabelEncoder()
        df_all['y'] = enc.fit_transform(df_all['patient_sex'])
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True)
        for train, val in sgkf.split(df_all['image_id'], df_all['y'], groups=df_all['patient_id']):
            train_ids = df_all['image_id'][train].to_frame()
            val_ids = df_all['image_id'][val].to_frame()
            train_ids = train_ids.rename(columns = {'image_id':0}).reset_index(drop=True)
            val_ids = val_ids.rename(columns = {'image_id':0}).reset_index(drop=True)
            # print(train_ids.head())
            folds.append([train_ids, val_ids])
        return folds
    else:
        kf = KFold(n_splits=k)
        for train_idx, val_idx in kf.split(df_train_val):
            # print(df_train_val.iloc[train_idx,:])
            folds.append([df_train_val.iloc[train_idx,:], df_train_val.iloc[val_idx,:]])
    return folds

def loadData(fold, df_all, batch_size, image_size):
    """Creates train and val loaders

    Parameters
    ----------
    fold : list of pandas.DataFrames
        [train, val]
    df_all : pandas.DataFrame
        Contains all available labeled data (original RANZCR-CLIP train.csv format)
    batch_size : int
        number of images per batch
    image_size : int
        size of images is image_size by image_size

    Returns
    -------
    train_loader : pytorch DataLoader
        loader for training set
    valid_loader : pytorch DataLoader
        loader for validation set

    """
    train_dataset = RANZCR_CLIP(df_all, fold[0], transform=get_transform(image_size, 'train'))
    train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle = True)    
    valid_dataset = RANZCR_CLIP(df_all, fold[1], transform=get_transform(image_size, 'val'))
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, valid_loader  

def loadTestData(df_test, df_all, batch_size, image_size):
    test_dataset = RANZCR_CLIP(df_all, df_test, transform=get_transform(image_size, 'val'))
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return test_loader

def loadMIMIC(df, batch_size, image_size):
    MIMIC_dataset = MIMIC_CXR(df, transform=get_transform(image_size, 'train'))
    MIMIC_loader = DataLoader(MIMIC_dataset, batch_size = batch_size, shuffle = True)
    return MIMIC_loader 

def loadRetinalData(fold, df_all, batch_size, image_size):
    """Creates train and val loaders

    Parameters
    ----------
    fold : list of pandas.DataFrames
        [train, val]
    df_all : pandas.DataFrame
        Contains all available labeled data (original RANZCR-CLIP train.csv format)
    batch_size : int
        number of images per batch
    image_size : int
        size of images is image_size by image_size

    Returns
    -------
    train_loader : pytorch DataLoader
        loader for training set
    valid_loader : pytorch DataLoader
        loader for validation set

    """
    train_dataset = RETINAL(df_all, fold[0], transform=get_transform(image_size, 'train'))
    train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle = True)    
    valid_dataset = RETINAL(df_all, fold[1], transform=get_transform(image_size, 'val'))
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, valid_loader 


def loadRetinalData2(df_all, batch_size, image_size, retinal_path, channel_avg, channel_std, split='train'):
    """Creates train and val loaders

    Parameters
    ----------
    fold : list of pandas.DataFrames
        [train, val]
    df_all : pandas.DataFrame
        Contains all available labeled data (original RANZCR-CLIP train.csv format)
    batch_size : int
        number of images per batch
    image_size : int
        size of images is image_size by image_size

    Returns
    -------
    train_loader : pytorch DataLoader
        loader for training set
    valid_loader : pytorch DataLoader
        loader for validation set

    """
    train_dataset = RETINAL(df_all, retinal_path, transform=get_transform(image_size, channel_avg, channel_std, split=split))
    train_loader = DataLoader(train_dataset, retinal_path, batch_size = batch_size , shuffle = True)    
    # valid_dataset = RETINAL(df_all, fold[1], transform=get_transform(image_size, 'val'))
    # valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    return train_loader

def loadTestRetinalData(df_test, df_all, batch_size, image_size):
    test_dataset = RETINAL(df_all, df_test, transform=get_transform(image_size, 'val'))
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return test_loader



# Transform

def get_transform(image_size, channel_avg, channel_std, split = 'train'):
    transforms_train = albumentations.Compose([
        albumentations.Crop(x_min=500, y_min=1000, x_max=3500, y_max=3000, always_apply=True),
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=channel_avg, std=channel_std, max_pixel_value=255.0),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.75),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        ], p=0.75),

        albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.75),
     #   CutoutV2(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.75),
    ])
    transforms_val = albumentations.Compose([
        albumentations.Crop(x_min=500, y_min=1000, x_max=3500, y_max=3000, always_apply=True),
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=channel_avg, std=channel_std, max_pixel_value=255.0),
    ])
    if split == 'train':
        return transforms_train
    elif split == 'val':
        return transforms_val
    else:
        raise NotImplementedError

def get_optim(model, optimizer, lr):
    if optimizer == 'Adam':
        optim = Adam(model.parameters(), lr=lr,weight_decay=lr/10)
    elif optimizer == 'SGD':
        optim = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise NotImplementedError
    
    return optim

def get_lossfn(loss,weights):
    if loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss == 'FocalLoss':
        criterion = FocalLoss(gamma=0.6, alpha=weights)
    else:
        raise NotImplementedError
    return criterion

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum() 

if __name__ == "__main__":
    # unit tests if we have time/if necessary
    pass
