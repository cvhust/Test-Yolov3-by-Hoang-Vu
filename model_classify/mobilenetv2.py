from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, ConfusionMatrixDisplay
import keras
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

check_point_dir = 'D:/Study/PythonPrj/pythonProject/TensorFlow-2.x-YOLOv3/model_classify/mobilenetv2.h5'


def __model__():
    base_model = MobileNetV2(input_shape=(224, 224, 3), input_tensor=None, include_top=False, weights='imagenet')
    base_model.trainable = True

    # define model
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(320, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(32, activation='softmax'))
    model.summary()

    model.compile(optimizer=Adam(epsilon=1e-08), loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(check_point_dir)
    return model
