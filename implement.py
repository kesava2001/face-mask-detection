import cv2
import tensorflow 
import numpy as np
from tensorflow import keras

model = keras.models.load_model('trained_model.h5')
label = ['Mask', 'No_Mask']

def call(img, model):
    image = cv2.resize(img, (224, 224))
    image = image.reshape(1, 224, 224, 3)
    image = np.array(image)
    res = model.predict(image)
    return label[np.argmax(res[0])]



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def roi(frame, face_cascade):
    faces = face_cascade.detectMultiScale(frame, scaleFactor=2, minNeighbors=8, flags=None, minSize=None, maxSize=None)


    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x,y), (x+w, y+h), (2, 8, 255), 1)        
        res = call(frame, model)

        cv2.putText(img, res, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 1)
        cv2.putText(frame, res, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame

def initialize():
    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()
        x = roi(frame, face_cascade)
        cv2.imshow('display', x)
        k = cv2.waitKey(1)
        if k == 27:
            break
    camera.release()
    cv2.destroyAllWindows()

initialize()