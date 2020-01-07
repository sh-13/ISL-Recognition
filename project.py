# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:00:56 2019

@author: sh_13
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(64, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(output_dim=64, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compiling CNN
classifier.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics = ['accuracy'])

# Fit CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model=classifier.fit_generator(training_set,
                        steps_per_epoch=200,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=20)

#Saving the model
import h5py
classifier.save('Trained_model.h5')

print(model.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()