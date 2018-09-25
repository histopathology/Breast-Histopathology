import glob
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.utils import to_categorical
from keras.models import load_model
from collections import defaultdict

def show_images(img_files):
    imgs = []
    
    for i in range(10):
        full_size_image = cv2.imread(img_files[i])
        imgs.append(cv2.resize(full_size_image, (50, 50), interpolation = cv2.INTER_CUBIC))
        
    for row in range(2):
        plt.figure(figsize=(20, 10))
        for col in range(5):
            if row == 0:
                plt.subplot(1,7,col+1)
                plt.imshow(imgs[col])
                plt.axis('off')
            else:
                plt.subplot(1,7,col+1)
                plt.imshow(imgs[5+col])
                plt.axis('off')
                
def create_cm(y_pred, y_actual):
    err = defaultdict(int)
    
    for p in range(len(y_pred)):
        if (y_pred[p][0] < y_pred[p][1]) and np.argmax(y_actual[p]) == 0:
            err["false_pos"] += 1
            
        elif (y_pred[p][0] > y_pred[p][1]) and np.argmax(y_actual[p]) == 1:
            err["false_neg"] += 1

        elif (y_pred[p][0] < y_pred[p][1]) and np.argmax(y_actual[p]) == 1:
            err["true_pos"] += 1
            
        elif (y_pred[p][0] > y_pred[p][1]) and np.argmax(y_actual[p]) == 0:
            err["true_neg"] += 1
          
    cm = [[err["true_neg"], err["false_pos"]], [err["false_neg"], err["true_pos"]]]
    return cm

def show_conf_matrix(cm):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Pastel1)
    classNames = ['IDC (-)','IDC (+)']
    plt.title('IDC (-) and IDC (+) Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]), horizontalalignment="center")
    plt.show()
	
def score_IDC(model, img_mean, patient_no):
    data_files = glob.glob("data/" + str(patient_no) + "/*/*.png")
    X = []
    Y = []

    for d in data_files:
        full_size_image = cv2.imread(d)
        X.append(cv2.resize(full_size_image, (50, 50), interpolation = cv2.INTER_CUBIC))
        if d.endswith("class0.png"):
            Y.append(0)
        else:
            Y.append(1)
            
    X = np.array(X, dtype=np.float64)
    X -= img_mean
    Y = to_categorical(Y)
    
    Y_pred = model.predict(X)
    pos = 0.0
    for p in range(len(Y_pred)):
        if Y_pred[p][0] < Y_pred[p][1]:
            pos += 1.0

    cm = create_cm(Y_pred, Y)
            
    score = pos / float(len(Y_pred))
    print("Patient: {} --> IDC aggressiveness score: {:0.4F}".format(patient_no, score))
    return score, cm