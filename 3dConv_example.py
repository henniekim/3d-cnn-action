from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import keras

import glob
import os
import argparse
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"

batch_size = 32


def generate_data(training_filename, training_label, batch_size, category_name):
    i = 0
    shuffled_indexes = np.arange(len(training_filename))

    while True:
        image_batch = []
        label_batch = []
        sample = ''
        np.random.shuffle(shuffled_indexes)
        i = 0

        for i in range(batch_size):
            sample = training_filename[shuffled_indexes[i]]
            frame_number = 0
            image_stack = []
            for frame_number in range(8):
                current_frame = 9 * frame_number + 10
                image = cv2.imread(source + '/' + sample + '/' + str(current_frame) + '.jpg')
                image = cv2.resize(image, dsize = (128, 128))

                image_stack.append(image)
            image_batch.append(image_stack)
            current_label = training_label[shuffled_indexes[i]]
            current_cls_number = np.where(category_name == current_label)
            current_category = to_categorical(current_cls_number[0], num_classes=200)
            label_batch.append(current_category)

        label = np.array(label_batch)
        reshaped_label = np.reshape(label, (batch_size, 200))

        yield np.array(image_batch), reshaped_label

'''
parser = argparse.ArgumentParser()

parser.add_argument("source_folder", type=str, help=" The folder which contains frames ")
parser.add_argument("training_info", type=str, help=" Training data information file (.csv) ")
parser.add_argument("validation_info", type=str, help=" Validation data information file (.csv) ")
parser.add_argument("categories", type=str, help=" Category file (.txt)")

args = parser.parse_args()

source = args.source_folder
csv_training = args.training_info
csv_validation = args.validation_info
txt_category = args.categories
'''

source = "/datahdd/Dataset/Moments_in_Time_Mini/processed"
format = "jpg"
txt_category = "/datahdd/Dataset/Moments_in_Time_Mini/moments_categories.txt"

csv_training = "/datahdd/Dataset/Moments_in_Time_Mini/trainingSet.csv"
csv_validation = "/datahdd/Dataset/Moments_in_Time_Mini/validationSet.csv"


vid_list = glob.glob(source+'/*/*/*.'+format)
num_vid = len(vid_list)

category_name = np.loadtxt(txt_category, delimiter=',', usecols=0, dtype=str, encoding='utf-8')

category_encoding = np.loadtxt(txt_category, delimiter=',', usecols=1, dtype=None, encoding='utf-8')

print('The number of training video is : '+str(num_vid))
print('The number of categories is : '+str(len(category_encoding)))

# Load information from csv (the file is supported by data provider)
training_filename = np.genfromtxt(csv_training, delimiter=',', usecols=0, dtype=None, encoding='utf-8')
training_label = np.genfromtxt(csv_training, delimiter=',', usecols=1, dtype=None, encoding='utf-8')

validation_filename = np.genfromtxt(csv_validation, delimiter=',', usecols=0, dtype=None, encoding='utf-8')
validation_label = np.genfromtxt(csv_validation, delimiter=',', usecols=1, dtype=None, encoding='utf-8')



model = Sequential()

model.add(Conv3D( filters = 64, kernel_size= (3, 3, 3), strides = (1,1,1) , padding ="same", activation="relu", data_format="channels_last", input_shape = (8, 128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling3D( pool_size = (1, 2, 2), strides=(2,2,2), padding = "same", data_format= None))

model.add(Conv3D( filters = 128, kernel_size= (3, 3, 3), strides= (1,1,1), padding= "same", activation="relu"))
model.add(MaxPooling3D( pool_size = (2, 2, 2), strides=(2,2,2), padding = "same", data_format= None))

model.add(Conv3D( filters = 256, kernel_size= (3, 3, 3), strides= (1,1,1), padding= "same", activation="relu"))
model.add(Conv3D( filters = 256, kernel_size= (3, 3, 3), strides= (1,1,1), padding= "same", activation="relu"))
model.add(MaxPooling3D( pool_size = (2, 2, 2), strides=(2,2,2), padding = "same", data_format= None))

model.add(Conv3D( filters = 512, kernel_size= (3, 3, 3), strides= (1,1,1), padding= "same", activation="relu"))
model.add(Conv3D( filters = 512, kernel_size= (3, 3, 3), strides= (1,1,1), padding= "same", activation="relu"))
model.add(MaxPooling3D( pool_size = (2, 2, 2), strides=(2,2,2), padding = "same", data_format= None))

model.add(Conv3D( filters = 512, kernel_size= (3, 3, 3), strides= (1,1,1), padding= "same", activation="relu"))
model.add(Conv3D( filters = 512, kernel_size= (3, 3, 3), strides= (1,1,1), padding= "same", activation="relu"))
model.add(MaxPooling3D( pool_size = (2, 2, 2), strides=(2,2,2), padding = "same", data_format= None))

model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense( units = 4096, activation="relu"))
model.add(Dense( units = 4096, activation="relu"))

model.add(Dense( units = 200, activation="softmax"))

model.summary()

#$x, y = generate_data(training_filename, training_label, batch_size, category_name)
sgd = optimizers.SGD(lr=.001, decay=1e-6, momentum=0.99, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose = 0, save_best_only = True, period = 1)
model.save('./3dConv'+'batchsize_'+str(batch_size))

model.fit_generator(generate_data(training_filename, training_label, batch_size, category_name), epochs =5 , verbose = 1, steps_per_epoch= len(training_filename) // batch_size, validation_data = generate_data(validation_filename, validation_label, batch_size, category_name), validation_steps = len(validation_filename) // (batch_size))

