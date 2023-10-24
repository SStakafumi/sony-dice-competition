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

random.seed(1)
warnings.filterwarnings('ignore')


# 定数設定
IMAGE_ONE_DICE_SIZE = 32 # 一つのサイコロの最終的な画像のサイズ
IMAGE_EXPAND_SIZE = 256 # 20x20pixから拡大処理(otsu resize)するサイズ
ONE_DICE_THRESHOLD = 12000

# 画像データ読み込み
X_train = np.load('/mnt/c/Users/user/MyData/SonyDice/X_train.npy')
X_train = np.reshape(X_train, [200000, 20, 20])

X_test = np.load('/mnt/c/Users/user/MyData/SonyDice/X_test.npy')
X_test = np.reshape(X_test, [10000, 20, 20])

# ラベルデータ読み込み
y_train = np.load('/mnt/c/Users/user/MyData/SonyDice/y_train.npy')
y_train.shape

def getRect(img):
    '''画像に対して矩形領域を検知する関数'''
    contours, hierarchy = cv2.findContours(image=img, # lanczosを使う
                                           mode=cv2.RETR_EXTERNAL, # 一番外側の輪郭のみ
                                           method=cv2.CHAIN_APPROX_SIMPLE) # 輪郭座標の詳細なし
    
    rect_center = []
    rect_size = []
    rect_angle = []
    rect_area = []

    for contour in contours:

        # 傾いた外接する矩形領域
        rect = cv2.minAreaRect(contour)
        
        # 検出した矩形の合計面積を求める
        tmp_area = cv2.contourArea(contour)
        rect_area.append(tmp_area)

        rect_center.append(rect[0])
        rect_size.append(rect[1])
        rect_angle.append(rect[2])
    
    return rect_center, rect_size, rect_angle, int(sum(rect_area))


def getOneDiceIndexAndRect():
    '''サイコロが一つの画像のindexとその矩形領域情報を返す
    Return:
        one_dice_index: インデックスのリスト
        contour_list: サイコロごとの輪郭情報
    '''

    one_dice_idx = []
    rect_list = []
    have_small_piece_img_num = 0

    for i in range(X_train.shape[0]):
        img = X_train[i, :]
        
        # resize dinarization
        img = cv2.resize(img, (IMAGE_EXPAND_SIZE, IMAGE_EXPAND_SIZE), interpolation=cv2.INTER_LANCZOS4)
        thresh, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        rect_center, rect_size, rect_angle, rect_sum_area = getRect(img)

        if rect_sum_area < ONE_DICE_THRESHOLD:
            # サイコロを一つだけ持つ画像のidxを記録
            one_dice_idx.append(i)
        else:
            # 画像内のサイコロの面積が閾値を超えていたら考えない
            continue

        # 目いっぱいclosing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)

        contours, _ = cv2.findContours(image=img, # lanczosを使う
                                       mode=cv2.RETR_EXTERNAL, # 一番外側の輪郭のみ
                                       method=cv2.CHAIN_APPROX_SIMPLE) # 輪郭座標の詳細なし
        dice_num = 0
        have_small_piece = False
        for contour in contours:
            
            # 傾いた外接する矩形領域
            rect = cv2.minAreaRect(contour)

            if (80 <= rect[1][0] <= 150) and (80 <= rect[1][1] <= 150):
                rect_list.append(rect)
                dice_num += 1
            else:
                have_small_piece = True

        if have_small_piece:
            have_small_piece_img_num += 1
            have_small_piece = False
                
        assert dice_num == 1, 'サイコロが0個もしくは2個以上含まれます'

    # print(have_small_piece_img_num)

    return one_dice_idx, rect_list


def getOneDiceImageInfoListFromRecWidth():
    '''矩形データのサイズからサイコロを一つだけ存在する画像データとそのインデックスを作成
    Reurun:
        imgs: サイコロ部分を切り取った224*224の画像
        img_labels: ラベル(目の数)
    '''
    # 処理したサイコロの画像を格納するリスト
    imgs = []

    # 画像のラベル(目の合計)をまとめたリスト
    labels = []

    # 想定外の画像(label>6)
    labels_unexp = []

    # サイコロが一つで割れているため検出できない場合
    labels_undetect = []

    for i in range(X_train.shape[0]):
    
        img = X_train[i, :]
        
        # resize and binarization 
        img = cv2.resize(img, (IMAGE_EXPAND_SIZE, IMAGE_EXPAND_SIZE), interpolation=cv2.INTER_LANCZOS4)
        _, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        rect_center, rect_size, rect_angle, rect_sum_area = getRect(img)

        # サイコロのrect情報を格納するリスト
        dice_rect = []
        for j, size_tmp in enumerate(rect_size):
            if (85 <= size_tmp[0] <= 150) and (85 <= size_tmp[1] <= 150):
                dice_rect.append(j)
        
        if dice_rect == []:
            labels_undetect.append(i)

        # サイコロの数が1つだったら学習データとしてカウント
        if len(dice_rect) == 1:
            center, size, angle = tuple(map(int, rect_center[dice_rect[0]])), tuple(map(int, rect_size[dice_rect[0]])), rect_angle[dice_rect[0]]

            # 面積の合計が閾値12000を超えていたらサイコロが2つ存在するので除外
            if rect_sum_area > ONE_DICE_THRESHOLD:
                print(f'two dice idx: {i}')
                continue 

            # 目盛りの合計値が6より大きいことはおかしい
            if y_train[i] > 6:
                labels_unexp.append(i)
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
        img = cv2.resize(img, (IMAGE_ONE_DICE_SIZE, IMAGE_ONE_DICE_SIZE), interpolation=cv2.INTER_LANCZOS4)

        imgs.append(img)

    return imgs, labels, labels_unexp, labels_undetect


def getOneDiceImageInfoListFromArea():
    '''面積からサイコロを一つだけ存在する画像データとそのインデックスを作成
    Reurun:
        idx: サイコロを一つのみ持つ画像のidx
        imgs: サイコロ部分を切り取ったの画像
        labels: ラベル(目の数)
    '''
    imgs = []
    labels = []
    one_dice_idx, rect_list = getOneDiceIndexAndRect()

    for i, img_idx in enumerate(one_dice_idx):
        img = X_train[img_idx, :]

        # resize, binarization
        img = cv2.resize(img, (IMAGE_EXPAND_SIZE, IMAGE_EXPAND_SIZE), interpolation=cv2.INTER_LANCZOS4)
        thresh, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        width, height = img.shape
        center, size, angle = rect_list[i]
        center, size = tuple(map(int, center)), tuple(map(int, size))

        # 変換行列
        trans = cv2.getRotationMatrix2D(center, angle, scale=1)
        
        # Rotation
        img = cv2.warpAffine(img, trans, (width, height))

        # Crop
        img = cv2.getRectSubPix(img, size, center)

        # resize lanczos
        img = cv2.resize(img, (IMAGE_ONE_DICE_SIZE, IMAGE_ONE_DICE_SIZE), interpolation=cv2.INTER_LANCZOS4)

        imgs.append(img.astype(np.uint8))
        labels.append(y_train[img_idx])

    return one_dice_idx, imgs, labels

def getOneDiceRotate90():
    '''90°に回転させて画像を4倍にかさましする関数
    Return:
        imgs: かさましした画像リスト
        labels: かさまししたラベルリスト
    '''
    _, imgs, labels = getOneDiceImageInfoListFromArea()

    imgs_x4 = []
    labels_x4 = []

    for i, img in enumerate(imgs):
        img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_rotate_90_counterclockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_rotate_180 = cv2.rotate(img, cv2.ROTATE_180)

        imgs_x4.append(img)
        imgs_x4.append(img_rotate_90_clockwise)
        imgs_x4.append(img_rotate_180)
        imgs_x4.append(img_rotate_90_counterclockwise)

        for j in range(4):
            labels_x4.append(labels[i])

    return imgs_x4, labels_x4


def myTransformer(img, label, data_type):
    if data_type == 'trn':
        img_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.73473), (0.42428))
        ])
    elif data_type == 'val':
        img_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.73464), (0.42431))
        ])
    
    img_t = img_transformer(img)
    # onehot encodingで返す
    label_t = torch.zeros(6, dtype=torch.long)
    label_t[int(label-1)] = 1.0

    return img_t, label_t



class ImageDataset(Dataset):
    '''Dataset クラス
    Attributes:
        data_type: Dataset のタイプ, 'trn', 'val', 'test'
    '''
    def __init__(self, data_type):
        self.data_type = data_type
        assert data_type in ['trn', 'val'], 'データタイプが想定外です'
        imgs, labels = getOneDiceRotate90()

        trn_imgs, val_imgs, trn_labels, val_labels = \
            train_test_split(imgs, labels, train_size=0.8, stratify=labels, random_state=1)
        
        if data_type == 'trn':
            self.imgs = trn_imgs
            self.labels = trn_labels
        elif data_type == 'val':
            self.imgs = val_imgs
            self.labels = val_labels
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        if self.data_type == 'trn':
            img = self.imgs[idx]
            img = img.astype(np.unit8)
            label = self.labels[idx]
            img_t, label_t = myTransformer(img, label, self.data_type)
        elif self.data_type == 'val':
            img = self.imgs[idx]
            img = img.astype(np.unit8)
            label = self.labels[idx]
            img_t, label_t = myTransformer(img, label, self.data_type)
        
        return img_t, label_t