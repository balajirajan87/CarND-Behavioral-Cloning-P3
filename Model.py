import os
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import sgd, adam
from keras.callbacks import Callback
import cv2
import numpy as np
import sklearn
from keras.models import Model
import matplotlib.pyplot as plt
from keras import metrics

samples = []
with open('./Datasets/Master_driving_log_new.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples_shuffled = shuffle(samples)

train_samples, validation_samples = train_test_split(samples_shuffled, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.4
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    source_len = len(source_path.split('/'))
                    filename = source_path.split('/')[4:source_len]
                    current_path = './'
                    for j in range(len(filename)):
                        if (j != len(filename)-1):
                            current_path += filename[j]
                            current_path += '/'
                        else:
                            current_path += filename[j]
                    image = cv2.imread(current_path)
                    images.append(image)
                    images.append(cv2.flip(image,1))
                steering_centre = float(batch_sample[3])
                angles.append(steering_centre)
                angles.append(steering_centre * (-1.0))
                steering_left = steering_centre + correction
                angles.append(steering_left)
                angles.append(steering_left * (-1.0))
                steering_right = steering_centre - correction
                angles.append(steering_right)
                angles.append(steering_right * (-1.0))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# when accuracy reaches ACCURACY_THRESHOLD
LOSS_THRESHOLD = 0.06

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_mean_squared_error') < LOSS_THRESHOLD):
            print("\nReached %2.2f%% Loss, so stopping training!!" %(LOSS_THRESHOLD*100))
            self.model.stop_training = True

callbacks = myCallback()
#define the model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,3,3,subsample=(2,2),activation="elu"))
model.add(Convolution2D(48,3,3,subsample=(2,2),activation="elu"))
model.add(MaxPooling2D(pool_size=(4, 4), strides=4))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50,activation="elu"))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam',metrics=[metrics.mean_squared_error])
model.fit_generator(train_generator,steps_per_epoch=np.ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps=np.ceil(len(validation_samples)/batch_size),epochs=20, verbose=1, callbacks=[callbacks])
print("saving the model")
model.save('model.h5')
exit()