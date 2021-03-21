import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp, crop_image1
from yolov3.configs import *
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator,load_img
from model_classify.mobilenetv2 import __model__
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# yolo = Load_Yolo_model()
model = __model__()
# image_path = "D:/test_image.PNG"
# origin_image = cv2.imread(image_path)
# image = crop_image1(yolo, image_path, "", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, resize=True, do_return=True)
# new_image = image.reshape((-1, 224, 224, 3)) / 255.0
# prediction = model.predict(new_image)
#
# ans = np.argmax(prediction) + 1
# # cv2.imshow("origin image", origin_image)
# cv2.imshow(f"gesture {ans}", origin_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

datagen = ImageDataGenerator(rescale=1/255)
test_generator = datagen.flow_from_directory(
        'D:/test_moi_detect',
        target_size=(224, 224),
        batch_size=64,
        shuffle=False,
        class_mode='categorical')

# loss, acc = model.evaluate(test_generator)
# print(loss)
# print(acc)
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
# # print('Confusion Matrix')
# # cf_matrix = confusion_matrix(y_test, y_pred)
# # print(cf_matrix)
# print('Classification Report')
# target_names = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
#                  '25', '26', '27', '28', '29', '30','31', '32']
# print(classification_report(test_generator.classes, y_pred))

cf_matrix = confusion_matrix(test_generator.classes, y_pred)
print(cf_matrix)
df_cm = pd.DataFrame(cf_matrix, index = test_generator.class_indices, dtype=int,
              columns = test_generator.class_indices)
plt.figure(figsize=(12,12))
sn.heatmap(df_cm, annot=True, cmap="OrRd", fmt='g')
