import os
import numpy as np
import cv2
import random
from sklearn.utils import shuffle
from yolov3.utils import image_preprocess1
import pickle
from sklearn.model_selection import train_test_split
import shutil
dirs = ["D:/Vu_Data_detect/", "D:/Hieu_Data_detect/", "D:/Manh_Data_detect/", "D:/Long_Data_detect/"]
X_train = []
y_train = []
X_test = []
y_test = []
X_val = []
y_val = []

classes = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5, '07': 6, '08': 7, '09': 8, '10': 9, '11': 10, '12': 11,
           '13': 12, '14': 13, '15': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19, '21': 20, '22': 21,
           '23': 22, '24': 23, '25': 24, '26': 25, '27': 26, '28': 27, '29': 28, '30': 29, '31': 30, '32': 31}


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


def write_data(X, y, dir):
    # dir = './train_data/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(X.shape[0]):
        image = X[i][0]
        name = X[i][1]
        gesture = get_key(y[i], classes)
        img_out_path = dir + gesture + '/' + name
        if not os.path.exists(dir + gesture):
            os.mkdir(dir + gesture)
        cv2.imwrite(img_out_path, image)


for dirt in dirs:
    for file in os.listdir(dirt):
        new_dir = dirt + file + '/'
        subject = file
        X = []
        y = []
        for image in os.listdir(new_dir):
            img_path = new_dir + image
            print(img_path)
            try:
                img = cv2.imread(img_path)
                # img = image_preprocess1(img, (224, 224, 3))
                X.append([img, image])
                y.append(classes[subject])
            except cv2.error as e:
                print("error")
                pass
        # ty le 70:15:15
        train_num = int(0.7 * len(X))
        test_num = int(0.85 * len(X))
        X_train.extend(X[:train_num])
        y_train.extend(y[:train_num])

        X_test.extend(X[train_num:test_num])
        y_test.extend(y[train_num:test_num])

        X_val.extend(X[test_num:])
        y_val.extend(y[test_num:])


# write train_data
write_data(X_train, y_train, 'D:/train_data/')
print("train done ")
# write test_data
write_data(X_test, y_test, 'D:/test_data/')
print("test done ")
# write val_data
write_data(X_val, y_val, 'D:/val_data/')
print("val done ")




