
# coding: utf-8

# In[1]:

import csv
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')

#creating an array with all the csv lines
lines = []
with open("data\driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

        
#I used the udacity sample data to train my model but since that wasnt enough I had to train further and the training done on
#my computer resulted in a different kind of line being updated in the driving_log.csv file so I used this command to edit the 
#'lines' list
for line in lines[8037:13396]:
    line[0] = "IMG/" + line[0].split("\\")[-1]
    line[1] = "IMG/" + line[1].split("\\")[-1]
    line[2] = "IMG/" + line[2].split("\\")[-1]


print(lines[8838][0]) 

#It must be noted very importantly that the first line in the reader are the column headings and must be avoided in the samples.
train_samples, validation_samples = train_test_split(lines[1:len(lines)], test_size=0.2)

batch_size = 8
correction_factor = 0.2

def generator(samples, batch_size = 16):
    global correction_factor
    while 1:
        
        
        #no_of_batches = len(samples)/batch_size
        shuffle(samples)
        num_samples = (len(samples)//batch_size)*batch_size
        for offset in range(0,num_samples, batch_size):
            batch_sample = samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            
            for line in batch_sample:
                
                centre_path = line[0]
                left_path = line[1]
                right_path = line[2]
                
                centre_path = ".//data//IMG//" + centre_path.split("/")[-1]
                left_path = ".//data//IMG//" + left_path.split("/")[-1]
                right_path = ".//data//IMG//" + right_path.split("/")[-1]
                
#               for every line in lines I appended the centre, left and right images to the 'images' array
#               and their corresponding steering angles along with the right corresponding steering angle
                centre_img = mpimg.imread(centre_path)
                left_img = mpimg.imread(left_path)
                right_img = mpimg.imread(right_path)

                images.append(centre_img)
                images.append(left_img)
                images.append(right_img)
                angle = float(line[3])
                #centre angle
                steering_angles.append(angle)
                #left angle
                steering_angles.append(angle + correction_factor)
                #right angle
                steering_angles.append(angle - correction_factor)
                
            X_train = np.array(images)
            Y_train = np.array(steering_angles)
            yield (X_train, Y_train)


train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


       
print("Training...")
model = Sequential()
#Normalizing Layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#Cropping the unnecessary parts of the image
model.add(Cropping2D(cropping = ((70,25),(0,0))))
#Convolutional Layers
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation(activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation(activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation(activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid'))
model.add(Activation(activation='relu'))
#Linear Layers
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch= (len(train_samples)//batch_size)*batch_size, validation_data=
                    validation_generator, nb_val_samples=(len(validation_samples)//batch_size)*batch_size, nb_epoch=3)
model.save('model.h5')
print("Training Complete")

