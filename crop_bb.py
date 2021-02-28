import os
from yolov3.dataset import Dataset
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *
from tensorflow.python.client import device_lib
import shutil
import numpy as np
import cv2
import tensorflow as tf
# from tensorflow.keras.utils import plot_model
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo, compute_loss
from yolov3.utils import load_yolo_weights, crop_image, crop_image1
from yolov3.configs import *
import PIL
from evaluate_mAP import get_mAP
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print(device_lib.list_local_devices())


dir = 'D:/Data_project/Long_Data/'
yolo = Load_Yolo_model()


if not os.path.exists('D:/Long_Data_detect/'):
    os.mkdir(f'D:/Long_Data_detect/')
t = 1
for gesture in os.listdir(dir):
    new_dir = dir + gesture + "/"
    out_dir = f"D:/Long_Data_detect/{gesture}/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for subject in os.listdir(new_dir):
        sub_path = new_dir + subject + '/'
        for img in os.listdir(sub_path):
            img_path = sub_path + img
            # print(img_path)
            try:
                temp = os.path.splitext(img)
                out_img_dir = out_dir + f"{temp[0]}^{gesture}{temp[1]}"
                print(out_img_dir)
                crop_image1(yolo, img_path, out_img_dir, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES)
                print(t)
                t += 1
            except PIL.UnidentifiedImageError as e:
                print("error")
                pass

# t = 1
#
# for item in os.listdir(dir):
#     new_dir = dir + item
#     out_dir = "D:/in/" + item
#     crop_image(yolo, new_dir, out_dir, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES)
#     print(t)
#     t += 1
