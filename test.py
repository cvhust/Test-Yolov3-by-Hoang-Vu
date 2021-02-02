
import os
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
subject = 'big_heart'
dir = f'D:/Data_test/{subject}/'
yolo = Load_Yolo_model()

os.mkdir(f'D:/Data_test/{subject}_detect')
t = 1
for item in os.listdir(dir):
    new_dir = dir + item
    detect_image(yolo, new_dir, f"D:/Data_test/{subject}_detect/{item}", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
    print(t)
    t += 1
