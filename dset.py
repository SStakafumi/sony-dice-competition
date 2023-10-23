import copy
import csv
import functools
import math
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings
import random
import cv2
from collections import namedtuple
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn as nn

from util import get_rect

random.seed(1)
warnings.filterwarnings('ignore')


# 定数設定
IMAGE_SIZE = (224, 224)

# 画像データ読み込み
X_train = np.load('/mnt/c/Users/user/MyData/SonyDice/X_train.npy')
X_train = np.reshape(X_train, [200000, 20, 20])

# ラベルデータ読み込み
y_train = np.load('/mnt/c/Users/user/MyData/SonyDice/y_train.npy')
y_train.shape


def getImageInfoList():
    '''サイコロを一つだけ存在する画像データとそのインデックスを作成
    Reurun:
        imgs: サイコロ部分を切り取った224*224の画像
        img_labels: ラベル(目の数)
    '''
    # 処理したサイコロの画像を格納するリスト
    imgs = []

    # 画像のラベル(目の合計)をまとめたリスト
    labels = []

    # 想定していない画像(label>6)
    labels_exception = []

    for i in range(X_train.shape[0]):
    
        img = X_train[i, :]
        
        # resize and binarization 
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        _, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        rect_center, rect_size, rect_angle = get_rect(img)
        # サイコロのrect情報を格納するリスト
        dice_rect = []
        for j, size_tmp in enumerate(rect_size):
            if (80 <= size_tmp[0] <= 150) and (80 <= size_tmp[1] <= 150):
                dice_rect.append(j)
        
        # サイコロの数が1つだったら学習データとしてカウント
        if len(dice_rect) == 1:
            center, size, angle = tuple(map(int, rect_center[dice_rect[0]])), tuple(map(int, rect_size[dice_rect[0]])), rect_angle[dice_rect[0]]
            if y_train[i] > 6:
                labels_exception.append(i)
                continue
            else:
                labels.append(y_train[i])
            
        else:
            continue

        width, height = img.shape

        # 変換行列
        trans = cv2.getRotationMatrix2D(center, angle, scale=1)
        
        # Rotation
        img = cv2.warpAffine(img, trans, (width, height))

        # Crop
        img = cv2.getRectSubPix(img, size, center)

        # resize lanczos
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)

        imgs.append(img)

    return imgs, labels, labels_exception

# class ImageDataset(Dataset):
#     def __init__(self, data_type):
#         if data_type == 'train':
#             imgs, labels = getImageInfoList()
