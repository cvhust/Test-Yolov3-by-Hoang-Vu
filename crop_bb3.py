import imutils
import cv2
import numpy as np
import os

dir = 'D:/27/'
out_dir = 'D:/27_detect/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for item in os.listdir(dir):
    temp = os.path.splitext(item)
    if temp[1] == '.txt':
        continue
    img_path = dir + item
    ano_path = dir + temp[0] + '.txt'
    img_out_path = out_dir + item
    _, bb_x, bb_y, bb_width, bb_height = open(ano_path).readline().strip().split()
    bb_x = float(bb_x)
    bb_y = float(bb_y)
    bb_height = float(bb_height)
    bb_width = float(bb_width)
    # read image
    image = cv2.imread(img_path)
    img_height, img_width, _ = image.shape
    bb_x, bb_y, bb_width, bb_height = bb_x * img_width, bb_y * img_height, bb_width * img_width, bb_height * img_height
    xmin, ymin = bb_x - bb_width / 2, bb_y - bb_height / 2
    xmax, ymax = bb_x + bb_width / 2, bb_y + bb_height / 2
    bb_img = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
    bb_img = imutils.resize(bb_img, width=200)
    cv2.imwrite(img_out_path, bb_img)
    print(img_path)

