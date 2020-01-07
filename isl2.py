# Model for Sign Language Recognition

# Importing all the libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(28, 28, 3)))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding extra convolution layers
classifier.add(Conv2D(16, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#classifier.add(Conv2D(256, kernel_size=2, activation='relu'))
#classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flatten
classifier.add(Flatten())

# Step 4 - Fully Connected Layer
classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(36, activation='softmax'))

# Compile the Model
classifier.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Keras Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)

#test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'ISL Gestures Dataset',
        target_size=(28, 28),
        batch_size=16,
        class_mode='binary')

#validation_generator = test_datagen.flow_from_directory(
#        'data/validation',
#        target_size=(150, 150),
#        batch_size=32,
#        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=7920,
        epochs=2)
import cv2
import numpy as np
import requests

cap = cv2.VideoCapture(0)
#fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    #fgmask = fgbg.apply(frame)
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    maskpredict = np.resize(mask,[1,200,200,3])
    classes = classifier.predict_classes(maskpredict)
    if(classes>9):
        cv2.putText(frame,chr(classes-10+ord('a')), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame,classes, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    
    k = cv2.waitKey(30) & 0xff
    if(k==27):
        break
    
cap.release()
cv2.destroyAllWindows()

#url = 'http://192.168.0.102:8080/shot.jpg'
#
#while True:
#    img_resp = requests.get(url)
#    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#    frame = cv2.imdecode(img_arr, -1)
#    
#    roi = frame[100:300, 100:300]
#    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 0)
#    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#    lower_skin = np.array([0,20,70], dtype=np.uint8)
#    upper_skin = np.array([20,255,255], dtype=np.uint8)
#    
#    mask = cv2.inRange(hsv, lower_skin, upper_skin)
#    maskpredict = np.resize(mask,[1,200,200,3])
#    cv2.putText(frame, "Testing", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#    #fgmask = fgbg.apply(img)
#    cv2.imshow("Android", frame)
#    cv2.imshow("fg", mask)
#    if(cv2.waitKey(1)==27):
#        break
#cv2.destroyAllWindows()