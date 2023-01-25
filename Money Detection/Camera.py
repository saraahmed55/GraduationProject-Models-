# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 22:29:49 2023

@author: EL-MAGD
"""

import os
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt

import zipfile

import tensorflow as tf
#import tensorflow_hub as hub
import math
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import shutil
from keras.regularizers import l2

import tensorflow as tf
import keras
model=tf.keras.models.load_model('C:\\Users\\EL-MAGD\\Desktop\\Money detection\\model\\earlyStopping - Copy.h5')

import cv2
import numpy as np
import tensorflow as tf
y = ['100',  '100', '10',  '10',  '200',  '200',  '20', '20',  '50',  '50',  '5', '5']
#import model
model=tf.keras.models.load_model('C:\\Users\\EL-MAGD\\Desktop\\Money detection\\model\\earlyStopping - Copy.h5',compile=False)

def Curr_Pred(path):
    img=cv2.imread(path)
    dim=(224,224)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    pred_probab = model.predict(img)[0]

    if max(pred_probab) < 0.45:
        print("Please take another picture")
    else:    
        pred_class = list(pred_probab).index(max(pred_probab))
        return y[list(pred_probab).index(max(pred_probab))]


#predection
#predec = Curr_Pred("C:/Users/EL-MAGD/Desktop/Money detection/dataset/valid")        
   

s=Curr_Pred("C:\\Users\\EL-MAGD\\Desktop\\Money detection\\m\\msg87991683-7519.jpg")
print(s)