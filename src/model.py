import numpy as np
import pandas as pd
import csv
import cv2
import numpy as np
import math
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Convolution2D


# Load csv file with log of the collected data during lap in the simulator

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split data in 80% training and 20% validation
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size = 0.2)


# extend the current data set with one extra entry
def extend (name, steering_angle, append_flipped = True, external_angle = 0):
    images_output = []
    angles_output = []
    image = cv2.imread(name)
    angle = steering_angle
    
    # append angle and image to the output
    images_output.append(image)
    angles_output.append(angle)

    if append_flipped == True:
        image = np.fliplr(image)
        angle = -angle
        images_output.append(image)
        if external_angle == 0:
            angles_output.append(angle)
        else:
            # angles_output.append(angle)
            # maybe the second one is correct
            # also flip the angles for correctness
            angles_output.append(-external_angle)

    return images_output, angles_output

'''
Instead of storing the preprocessed data in memory all at once, we create 
a generator which pulls pieces of the data and process them on the fly only 
when you need them, which is much more memory-efficient
'''
def generator(samples, batch_size = 32, multiple_cameras = True, append_flipped = True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../data/IMG/'+batch_sample[0].split('/')[-1]
                steering_center = float(batch_sample[3])
                center_image, center_angle = extend(name, steering_center, append_flipped = append_flipped)

                images.append(center_image)
                angles.append(center_angle)

                # if we want to use multiple cameras, then also append this to images and angles
                if multiple_cameras == True:
                    # create adjusted steering measurements for the side camera images
                    correction = 0.2 # this is a parameter to tune
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction

                    #load left data
                    left_name = '../data/IMG/'+batch_sample[1].split('/')[-1]
                    left_image, left_angle = extend(left_name, steering_left, append_flipped = append_flipped, external_angle = steering_right)
                    
                    #load right data
                    right_name = '../data/IMG/'+batch_sample[2].split('/')[-1]
                    right_image, right_angle = extend(right_name, steering_right, append_flipped = append_flipped, external_angle = steering_left)

                    #append left information to training set
                    images.append(left_image)
                    angles.append(left_angle)

                    #append right info to data
                    images.append(right_image)
                    angles.append(right_angle)
                    
            # Define new training data, shuffle and return batch
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Set our batch size
batch_size = 42 #
n_epochs = 1
# compile and train the model using the generator function
# The existing data is augmented by adding the flipped images and also data from the left and right camera
train_generator = generator(train_samples, batch_size = batch_size, multiple_cameras = True, append_flipped = True)
validation_generator = generator(validation_samples, batch_size = batch_size, multiple_cameras = True, append_flipped = True)


original_row = 160
original_col = 320
crop_top = 70
crop_bottom = 25 
ch, row, col = 3, original_row - crop_top - crop_bottom, original_col  # Trimmed image format

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(ch, original_row, original_col)))
#Trimming layer
model.add(Cropping2D(cropping = ((crop_top, crop_bottom), (0, 0))))

# Define loss function and optimizer
model.compile(loss='mse', optimizer='adam')


# Define how to use data
model.fit_generator(train_generator, 
            steps_per_epoch = math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps = math.ceil(len(validation_samples)/batch_size), 
            epochs = n_epochs, verbose = 1)
