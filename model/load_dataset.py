import os
import random
import cv2 as cv
import numpy as np
from tqdm import tqdm

IMG_SIZE = 128
NORMAL_LABEL = -1
PNEUMONIA_LABEL = 1

""""
 - đọc ảnh
 - grayscale
 - resize 128 x 128
 - flatten thành vector
 - lưu vào images
 - lưu nhãn
"""
def collect(split: str = "train", base_dir: str = "chest_xray"):
    normal = "NORMAL"
    pneu = "PNEUMONIA"
    images = []
    labels = []

    # normal = -1
    normal_dir = os.path.join(base_dir, split, normal)

    for img_file in tqdm(os.listdir(normal_dir), desc=f"Loading {split}/NORMAL"):
        img_path = os.path.join(normal_dir, img_file)
        img = cv.imread(img_path)
        if img is None:
            print("Không thể đọc ảnh:", img_path)
            continue
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT)
        img = img.reshape(-1)
        images.append(img)
        labels.append(-1)

    # pneu = 1
    pneu_dir = os.path.join(base_dir, split, pneu)
    for img_file in tqdm(os.listdir(pneu_dir), desc=f"Loading {split}/PNEUMONIA"):
        img_path = os.path.join(pneu_dir, img_file)
        img = cv.imread(img_path)
        if img is None:
            print("Không thể đọc ảnh:", img_path)
            continue
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT)
        img = img.reshape(-1)
        images.append(img)
        labels.append(1)

    X = np.stack(images, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    data = list(zip(X, y))
    random.shuffle(data)
    X, y = zip(*data)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

class Scaler:
    """""
    Chuẩn hóa các đặc trưng bằng giá trị trung bình và độ lệch chuẩn của tập train
    Quan trọng:
        - fit_transform() chỉ được sử dụng trên tập train
        - transform() được sử dụng trên tập validation và tập test
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        X_scaled = (X - self.mean) / (self.std + 1e-8)

        return X_scaled

    def transform(self, X: np.ndarray):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        X_scaled = (X - self.mean) / (self.std + 1e-8)

        return X_scaled
