import random
import numpy as np
import warnings
import random
import cv2
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from torchvision import transforms

import torch
import torch.cuda
from torch.utils.data import Dataset
from torch import nn as nn

random.seed(1)
warnings.filterwarnings('ignore')


# 定数設定
IMAGE_ONE_DICE_SIZE = 32 # 一つのサイコロの最終的な画像のサイズ
IMAGE_EXPAND_SIZE = 256 # 20x20pixから拡大処理(otsu resize)するサイズ
THRESHOLD_BETWEEN12_TRAIN= THRESHOLD_BETWEEN12_DENOISED = 12000 # 訓練データのサイコロ１つと２つの面積の閾値
THRESHOLD_BETWEEN12 = 13000 
THRESHOLD_BETWEEN23= 23000 
THRESHOLD_BETWEEN23_DENOISED = 20000 # _DENOISEはUNetであらかじめ処理したもの
TWO_DICE_IDX = [15654, 19878, 21817, 22329]
THREE_DICE_IDX = [1995, 3422, 12838, 21186, 21866, 24648, 7393, 11275, 11613, 12234, 13481, 23319, 24588,
                  362, 614, 1203, 1355, 5042, 5144, 6027, 7332, 8318, 8616, 12503, 12676, 15231, 16003, 
                  16796, 20206, 22462, 23387, 23756, 24323]
NOISE_THRESHOLD = 2500
# TEST_DATA_SIZE = 1000 # テスト用全部読み込んでると時間がもったいないため

# 画像データ読み込み
X_train = np.load('/mnt/c/Users/user/MyData/SonyDice/X_train.npy')
X_train = np.reshape(X_train, [200000, 20, 20])
# X_train = X_train[:TEST_DATA_SIZE, :]

# 10/23に更新されたデータを読み取る
X_test = np.load('/mnt/c/Users/user/MyData/SonyDice/X_test_renew.npy')
X_test = np.reshape(X_test, [-1, 20, 20])

# UNetでデノイズされたデータを読み取る
X_test_denoised = np.load('/mnt/c/Users/user/MyData/SonyDice/X_test_denoised_by_UNet.npy')

# ラベルデータ読み込み
y_train = np.load('/mnt/c/Users/user/MyData/SonyDice/y_train.npy')
# y_train = y_train[:TEST_DATA_SIZE]

def getDiceSumArea(img, data_type):
    '''入力された画像に含まれるサイコロの面積の総和を返す関数 (testデータに対してはノイズを除去)
    
    args: 
        img(ndarray): resize->binarizationされた画像
        dtata_type(str): trn, val, test, test_denoisedのどれか
    
    Returns:
        sum_area_dice(int): 画像に含まれるサイコロの面積の総和
        '''

    assert data_type in ['trn', 'val', 'test', 'test_denoised'], 'データタイプを確認してください'

    contours, _ = cv2.findContours(image=img,
                                   mode=cv2.RETR_EXTERNAL, # 一番外側の輪郭のみ
                                   method=cv2.CHAIN_APPROX_SIMPLE) # 輪郭座標の詳細なし
    
    rect_area = []
    for contour in contours:  
        # 検出した矩形の合計面積を求める
        tmp_area = cv2.contourArea(contour)
        
        if data_type in ['trn', 'val', 'test_denoised']:
            rect_area.append(tmp_area)
        # testデータではノイズを考えるため, ノイズの閾値よりも面積が大きければ面積リストに追加する
        elif data_type == 'test':
            if tmp_area > NOISE_THRESHOLD:
                rect_area.append(tmp_area)

    # カウントされた面積の合計を出す
    sum_area_dice = int(sum(rect_area))
    
    return sum_area_dice


def getIndexForEachDice(data_type):
    '''サイコロの種類によってそれぞれのインデックスを返す関数
    Returne:
        one_dice_idx : 1つのサイコロを持つ画像のインデックス
        two_dice_idx : 2つのサイコロを持つ画像のインデックス
        three_dice_idx : 3つのサイコロを持つ画像のインデックス
    '''

    assert data_type in ['trn', 'test', 'test_denoised'], 'データタイプを確認してください'

    one_dice_idx = []
    two_dice_idx = []
    three_dice_idx = []

    if data_type == 'trn':
        X = X_train

        for i in range(X.shape[0]):
            img = X[i, :]

            # denoise -> resize(lanczos4)
            img = cv2.resize(img, (IMAGE_EXPAND_SIZE, IMAGE_EXPAND_SIZE), interpolation=cv2.INTER_LANCZOS4)
            thresh, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 画像に含まれるサイコロの面積の総和を求める
            sum_area_dice = getDiceSumArea(img, data_type)

            if sum_area_dice < THRESHOLD_BETWEEN12_TRAIN:
                # サイコロを一つだけ持つ画像のidxを記録
                one_dice_idx.append(i)
            else:
                two_dice_idx.append(i)

    elif data_type == 'test':
        X = X_test

        for i in range(X.shape[0]):
            img = X[i, :]

            # denoise -> resize(lanczos4) -> dinarization
            img = cv2.fastNlMeansDenoising(src=img, h=30, templateWindowSize=3, searchWindowSize=7)
            img = cv2.resize(img, (IMAGE_EXPAND_SIZE, IMAGE_EXPAND_SIZE), interpolation=cv2.INTER_LANCZOS4)
            thresh, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 画像に含まれるサイコロの面積の総和を求める
            sum_area_dice = getDiceSumArea(img, data_type)
            
            # もし総面積13000以下で, かつサイコロ２個の例外にも当てはまらなかったらサイコロ１つとみなす
            if (sum_area_dice <= THRESHOLD_BETWEEN12) and (i not in TWO_DICE_IDX):
                one_dice_idx.append(i)
            # もし総面積が13000より大きく23000以下でサイコロ3個の例外にも当てはまらなかったらサイコロ２つとみなす
            elif (THRESHOLD_BETWEEN12 < sum_area_dice <= THRESHOLD_BETWEEN23) and (i not in THREE_DICE_IDX):
                two_dice_idx.append(i)
            elif sum_area_dice > THRESHOLD_BETWEEN23:
                three_dice_idx.append(i)
            
        for idx in TWO_DICE_IDX:
            two_dice_idx.append(idx)

        for idx in THREE_DICE_IDX:
            three_dice_idx.append(idx)

        # sort
        two_dice_idx = sorted(two_dice_idx)
        three_dice_idx = sorted(three_dice_idx)

    elif data_type == 'test_denoised':
        X = X_test_denoised

        for i in range(X.shape[0]):
            img = X[i, :]

            # すでにデノイズされているためここではcv2.fastNlMeansDenoisingはしない
            # denoise -> resize(lanczos4)
            img = cv2.resize(img, (IMAGE_EXPAND_SIZE, IMAGE_EXPAND_SIZE), interpolation=cv2.INTER_LANCZOS4)
            thresh, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 画像に含まれるサイコロの面積の総和を求める
            sum_area_dice = getDiceSumArea(img, data_type)

            if sum_area_dice < THRESHOLD_BETWEEN12_DENOISED:
                # サイコロを一つだけ持つ画像のidxを記録
                one_dice_idx.append(i)
            elif sum_area_dice < THRESHOLD_BETWEEN23_DENOISED:
                two_dice_idx.append(i)
            else:
                three_dice_idx.append(i)

    # trainデータの場合はthree_dice_idxは空
    return one_dice_idx, two_dice_idx, three_dice_idx


def getOneDiceImage(idx, data_type, get_label=False):
    '''一つのサイコロを持つ画像のリストを返す関数.テストデータの場合デノイズ処理される.
    Return:
        imgs_list: 分離された画像(ndarray)のリスト (20*20)'''

    assert data_type in ['trn', 'test', 'test_denoised'], '画像のタイプを確認してください'

    imgs_list = []
    labels_list = []

    if data_type == 'trn':
        X = X_train
    elif data_type == 'test':
        X = X_test
    elif data_type == 'test_denoised':
        X = X_test_denoised
    
    for i in idx:
        img = X[i, :]

        # テストデータだったらデノイズを行う
        if data_type == 'test':
            img = cv2.fastNlMeansDenoising(src=img, h=30, templateWindowSize=3, searchWindowSize=7)
        
        imgs_list.append(img)

        if get_label:
            labels_list.append(y_train[i])

    if get_label:
        return imgs_list, labels_list
    else:
        return imgs_list



def devideTwoImage(idx, data_type):
    '''2つのサイコロを持つ画像を2つの画像に分離し, 画像をリストとして返す関数. テストデータの場合デノイズ処理される.

    Return:
        imgs_list: 分離された画像(ndarray)のリスト, 長さは元の2倍, (20*20)'''
    
    assert data_type in ['trn', 'test', 'test_denoised'], '画像のタイプを確認してください'

    imgs_list = []

    if data_type == 'trn':
        X = X_train
    elif data_type == 'test':
        X = X_test
    elif data_type == 'test_denoised':
        X = X_test_denoised

    for i in idx:
        img = X[i, :]

        # テストデータだったらデノイズを行う
        if data_type == 'test':
            img = cv2.fastNlMeansDenoising(src=img, h=30, templateWindowSize=3, searchWindowSize=7)

        dice_pix = []
        for j in range(20):
            for k in range(20):
                if img[j, k] >= 50:
                    dice_pix.append([j, k])
        dice_pix = np.array(dice_pix)
        
        # spectral clustering
        clustering = SpectralClustering(n_clusters=2, 
                                        assign_labels='discretize', 
                                        random_state=1, 
                                        affinity='nearest_neighbors').fit(dice_pix)
        
        # 一つ目の画像
        img_copy = img.copy()
        for l, [x, y] in enumerate(dice_pix):
            label = clustering.labels_[l]
            if label == 0:
                img_copy[x, y] = 1
        imgs_list.append(img_copy)

        # 二つ目の画像
        img_copy = img.copy()
        for m, [x, y] in enumerate(dice_pix):
            label = clustering.labels_[m]
            if label == 1:
                img_copy[x, y] = 1
        imgs_list.append(img_copy)

    return imgs_list


def devideThreeImage(idx, data_type):
    '''3つのサイコロを持つ画像を3つの画像に分離し, 画像をリストとして返す関数
    Return:
        imgs_list: デノイズ処理された分離された画像(ndarray)のリスト, 長さは元の3倍, (20*20)'''
    
    assert data_type in ['test', 'test_denoised'], 'この関数はテストデータしか受け付けません'

    imgs_list = []

    if data_type == 'test':
        X = X_test
    elif data_type == 'test_denoised':
        X = X_test_denoised

    for i in idx:
        img = X[i, :]

        # デノイズ
        if data_type == 'test':
            img = cv2.fastNlMeansDenoising(src=img, h=30, templateWindowSize=3, searchWindowSize=7)

        dice_pix = []
        for j in range(20):
            for k in range(20):
                if img[j, k] >= 50:
                    dice_pix.append([j, k])
        dice_pix = np.array(dice_pix)
        
        # spectral clustering
        clustering = SpectralClustering(n_clusters=3, 
                                        assign_labels='discretize', 
                                        random_state=2, 
                                        affinity='nearest_neighbors').fit(dice_pix)
        
        # 一つ目の画像
        img_copy = img.copy()
        for l, [x, y] in enumerate(dice_pix):
            label = clustering.labels_[l]
            if label == 0 or label == 1:
                img_copy[x, y] = 1
        imgs_list.append(img_copy)

        # 二つ目の画像
        img_copy = img.copy()
        for m, [x, y] in enumerate(dice_pix):
            label = clustering.labels_[m]
            if label == 1 or label == 2:
                img_copy[x, y] = 1
        imgs_list.append(img_copy)

        # 三つ目の画像
        img_copy = img.copy()
        for n, [x, y] in enumerate(dice_pix):
            label = clustering.labels_[n]
            if label == 2 or label == 0:
                img_copy[x, y] = 1
        imgs_list.append(img_copy)

    return imgs_list


def devideImage(data_type, one_dice_only=True):
    '''複数のサイコロを持つ画像に対して, サイコロを2つないしは3つの画像に分離し, 
    全画像を連結したうえで, リストとして返す関数

    Return:
        imgs_list: 全画像のリスト trainの場合 -> サイコロ1つの画像リスト + サイコロ2つの画像を分離し長さ2倍になった画像リスト
        20*20の画像である
    '''
    
    if data_type == 'trn':
        one_dice_idx, two_dice_idx, _ = getIndexForEachDice(data_type)
        if one_dice_only:
            imgs_list = getOneDiceImage(one_dice_idx, data_type)
        else:
            imgs_list = getOneDiceImage(one_dice_idx, data_type) \
                + devideTwoImage(two_dice_idx, data_type)

    elif data_type == 'test':
        one_dice_idx, two_dice_idx, three_dice_idx = getIndexForEachDice(data_type)
        imgs_list = getOneDiceImage(one_dice_idx, data_type) \
            + devideTwoImage(two_dice_idx, data_type) \
            + devideThreeImage(three_dice_idx, data_type)
    
    elif data_type == 'test_denoised':
        one_dice_idx, two_dice_idx, three_dice_idx = getIndexForEachDice(data_type)
        imgs_list = getOneDiceImage(one_dice_idx, data_type) \
            + devideTwoImage(two_dice_idx, data_type) \
            + devideThreeImage(three_dice_idx, data_type)
    
    return imgs_list

    

def getOneDiceRectanglar(img, data_type):
    '''サイコロが一つの画像の矩形領域情報を返す

    args:
        img(ndarray): 20*20の画像(testデータの場合デノイズされたものを入れる)
        data_type(str): trn, test, test_denoised

    Return:
        contour(list): サイコロの輪郭情報, [(center_y, center_x), (height, width), angle]
        mt_2_dice(bool): 画像内にサイコロが2個以上検出された時True
        zero_dice(bool): 画像内にサイコロが検出されなかった時True
    '''

    assert data_type in ['trn', 'test', 'test_denoised']

    # resize binarization
    img = cv2.resize(img, (IMAGE_EXPAND_SIZE, IMAGE_EXPAND_SIZE), interpolation=cv2.INTER_LANCZOS4)
    thresh, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 目いっぱいclosing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)

    # データタイプがtestだったら目いっぱいopening
    if data_type == 'test':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)

    contours, _ = cv2.findContours(image=img, # lanczosを使う
                                mode=cv2.RETR_EXTERNAL, # 一番外側の輪郭のみ
                                method=cv2.CHAIN_APPROX_SIMPLE) # 輪郭座標の詳細なし
    
    dice_num = 0
    dice_rects = []
    unexp_rects = []

    for contour in contours:
        # 傾いた外接する矩形領域
        rect = cv2.minAreaRect(contour)

        if (80 <= rect[1][0] <= 150) and (80 <= rect[1][1] <= 150):
            dice_rects.append(rect)
            dice_num += 1
        else:
            unexp_rects.append(rect)
    
    # debug
    mt2_dice = False
    zero_dice = False
    if dice_num >= 2:
        mt2_dice = True
    if dice_num == 0:
        zero_dice = True
    
    if data_type == 'trn':
        assert dice_num == 1, '検出されたサイコロの数が0, もしくは2個以上あります.'
        dice_rect = dice_rects[0]
    
    elif data_type in ['test', 'test_denoised']:
        if len(dice_rects) == 1:
            dice_rect = dice_rects[0]
        # 2個以上あったら矩形の面積が大きいほう
        elif len(dice_rects) >= 2:
            max_area = -1
            for i, tmp in enumerate(dice_rects):
                w = tmp[1][0]
                h = tmp[1][1]
                # 矩形の面積
                tmp_area = w * h
                if tmp_area > max_area:
                    dice_rect = dice_rects[i]
        # 0個だったらノイズと判断されたやつから面積が一番大きいやつ
        elif len(dice_rects) == 0:
            max_area = -1
            for i, tmp in enumerate(unexp_rects):
                w = tmp[1][0]
                h = tmp[1][1]
                # 矩形の面積
                tmp_area = w * h
                if tmp_area > max_area:
                    dice_rect = unexp_rects[i]

    return *dice_rect, mt2_dice, zero_dice


def getCroppedImgInfoList(data_type):
    '''切り取られたサイコロの画像の情報を返す関数, 
    trainデータの場合, サイコロが一つの画像のみでlabelも返す
    testデータの場合, 全データを連結したリストを返す
    
    Return:
        imgs: サイコロ部分を切り取った画像のリスト
        labels: サイコロのラベル(目), trainデータのみ    
    '''

    assert data_type in ['trn', 'test', 'test_denoised']

    imgs_output = []
    
    if data_type == 'trn':
        # 20*20の画像のリスト, one_dice_onlyで学習データにはサイコロ一つしか考えない
        one_dice_idx, _, _ = getIndexForEachDice(data_type='trn')
        imgs, labels = getOneDiceImage(idx=one_dice_idx, data_type='trn', get_label=True)

        for img in imgs:
            
            # 20*20 の画像に対して
            center, size, angle, _, _ = getOneDiceRectanglar(img, data_type)
            center, size = tuple(map(int, center)), tuple(map(int, size))

            # resize, binarization
            img = cv2.resize(img, (IMAGE_EXPAND_SIZE, IMAGE_EXPAND_SIZE), interpolation=cv2.INTER_LANCZOS4)
            thresh, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            width, height = img.shape

            # 変換行列
            trans = cv2.getRotationMatrix2D(center, angle, scale=1)
            
            # Rotation
            img = cv2.warpAffine(img, trans, (width, height))

            # Crop
            img = cv2.getRectSubPix(img, size, center)

            # resize lanczos
            img = cv2.resize(img, (IMAGE_ONE_DICE_SIZE, IMAGE_ONE_DICE_SIZE), interpolation=cv2.INTER_LANCZOS4)

            imgs_output.append(img.astype(np.uint8))

        return imgs_output, labels
    
    if data_type in ['test', 'test_denoised']:
        # デノイズ処理され分離され画像内に一つしかサイコロがない状態となった画像リスト
        imgs = devideImage(data_type)

        mt2_dice_num = []
        zero_dice_num = []

        for img in imgs:   
            # 20*20 の画像に対して
            center, size, angle, mt2_dice, zero_dice = getOneDiceRectanglar(img, data_type)
            center, size = tuple(map(int, center)), tuple(map(int, size))
            mt2_dice_num.append(mt2_dice)
            zero_dice_num.append(zero_dice)

            # resize, binarization
            img = cv2.resize(img, (IMAGE_EXPAND_SIZE, IMAGE_EXPAND_SIZE), interpolation=cv2.INTER_LANCZOS4)
            thresh, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            width, height = img.shape

            # 変換行列
            trans = cv2.getRotationMatrix2D(center, angle, scale=1)
            
            # Rotation
            img = cv2.warpAffine(img, trans, (width, height))

            # Crop
            img = cv2.getRectSubPix(img, size, center)

            # resize lanczos
            img = cv2.resize(img, (IMAGE_ONE_DICE_SIZE, IMAGE_ONE_DICE_SIZE), interpolation=cv2.INTER_LANCZOS4)

            imgs_output.append(img.astype(np.uint8))

        # debug
        print(f'サイコロが2つ以上あると判定された画像の枚数: {sum(mt2_dice_num)}')
        print(f'サイコロが0個であると判定された画像の枚数: {sum(zero_dice_num)}')

        return imgs_output

def getOneDiceRotate90(data_type):
    '''90°に回転させて画像を4倍にかさましする関数
    Return:
        imgs: かさましした画像リスト
        labels: かさまししたラベルリスト
    '''
    imgs, labels = getCroppedImgInfoList(data_type='trn')

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

        if data_type == 'train':
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
    label_t = torch.tensor(label-1, dtype=torch.long)

    return img_t, label_t


class ImageDataset(Dataset):
    '''Dataset クラス
    Attributes:
        data_type: Dataset のタイプ, 'trn', 'val'
    '''
    def __init__(self, data_type):
        self.data_type = data_type
        assert data_type in ['trn', 'val'], 'データタイプが想定外です'
        # 与えられた訓練データ(200000枚から選んだもの)
        imgs, labels = getOneDiceRotate90(data_type='train')

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
        img = self.imgs[idx]
        img = img.astype(np.uint8)
        label = self.labels[idx]
        img_t, label_t = myTransformer(img, label, self.data_type)
        
        return img_t, label_t