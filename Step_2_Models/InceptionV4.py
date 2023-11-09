# Author: Jiasong Liu
import os
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, BatchNormalization, Dropout, Concatenate

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import uuid
from datetime import datetime


class LrChanging(tf.keras.callbacks.Callback):
    # doing a little control about loss here, if the loss presist, we divide learning rate by 10
    global origin_rate
    origin_rate = 0.045
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr.read_value()
        global origin_rate
        if epoch != 0 and epoch % 10 == 0:
            lr = self.model.optimizer.lr.assign(origin_rate*(0.94**epoch))
        print("Current lr:" + str(lr.numpy()))

# The basic convolution unit, doing a Conv2D then Batchnormalization
def convolution_unit(x, filters, kernel=(3, 3), bn_axis = 3, padding = 'same', strides=(1, 1)):
    x = Conv2D(filters, kernel, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    return x

# the stem module of InceptionV4, implement as the structure given by paper
def stem(input, concat_axis = 3):
    # first route
    x = convolution_unit(input, 32, strides=(2, 2), padding = 'valid')
    x = convolution_unit(x, 32, padding = 'valid')
    x = convolution_unit(x, 64)
    # branch 1
    x_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x_2 = convolution_unit(x, 96, strides=(2, 2), padding='valid')
    # merge 1
    x = Concatenate(axis = concat_axis)([x_1, x_2])
    # branch 2
    x_1 = convolution_unit(x, 64, kernel=(1, 1))
    x_1 = convolution_unit(x_1, 96, padding='valid') 

    x_2 = convolution_unit(x, 64, kernel=(1, 1))
    x_2 = convolution_unit(x_2, 64, kernel=(1, 7))
    x_2 = convolution_unit(x_2, 64, kernel=(7, 1))
    x_2 = convolution_unit(x_2, 96, padding='valid')
    # merge 2
    x = Concatenate(axis = concat_axis)([x_1, x_2])
    # branch 3
    x_1 = convolution_unit(x, 192, strides=(2, 2), padding='valid')
    x_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    # merge 3
    x = Concatenate(axis = concat_axis)([x_1, x_2])
    return x

# the inception-A module
def inception_A_mod(input, concat_axis = 3):
    # sub_module_1
    x_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    x_1 = convolution_unit(x_1, 96, kernel=(1, 1))
    # sub_module_2
    x_2 = convolution_unit(input, 96, kernel=(1, 1))
    # sub_module_3
    x_3 = convolution_unit(input, 64, kernel=(1, 1))
    x_3 = convolution_unit(x_3, 96)
    # sub_module_4
    x_4 = convolution_unit(input, 64, kernel=(1, 1))
    x_4 = convolution_unit(x_4, 96)
    x_4 = convolution_unit(x_4, 96)
    # merge
    x = Concatenate(axis = concat_axis)([x_1, x_2, x_3, x_4])
    return x

# the inception-B module
def inception_B_mod(input, concat_axis = 3):
    # sub_module_1
    x_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    x_1 = convolution_unit(x_1, 128, kernel=(1, 1))
    # sub_module_2
    x_2 = convolution_unit(input, 384, kernel=(1, 1))
    # sub_module_3
    x_3 = convolution_unit(input, 192, kernel=(1, 1))
    x_3 = convolution_unit(x_3, 224, kernel=(7, 1))
    x_3 = convolution_unit(x_3, 256, kernel=(1, 7))
    # sub_module_4
    x_4 = convolution_unit(input, 192, kernel=(1, 1))
    x_4 = convolution_unit(x_4, 192, kernel=(1, 7))
    x_4 = convolution_unit(x_4, 224, kernel=(7, 1))
    x_4 = convolution_unit(x_4, 224, kernel=(1, 7))
    x_4 = convolution_unit(x_4, 256, kernel=(7, 1))
    # merge
    x = Concatenate(axis = concat_axis)([x_1, x_2, x_3, x_4])
    return x

# the inception-C module
def inception_C_mod(input, concat_axis = 3):
    # sub_module_1
    x_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    x_1 = convolution_unit(x_1, 256, kernel=(1, 1))
    # sub_module_2
    x_2 = convolution_unit(input, 256, kernel=(1, 1))
    # sub_module_3 with 2 branches
    x_3 = convolution_unit(input, 384, kernel=(1, 1))
    x_3_1 = convolution_unit(x_3, 256, kernel=(1, 3))
    x_3_2 = convolution_unit(x_3, 256, kernel=(3, 1))
    # sub_module 4
    x_4 = convolution_unit(input, 384, kernel=(1, 1))
    x_4 = convolution_unit(x_4, 448, kernel=(1, 3))
    x_4 = convolution_unit(x_4, 512, kernel=(3, 1))
    x_4_1 = convolution_unit(x_4, 256, kernel=(3, 1))
    x_4_2 = convolution_unit(x_4, 256, kernel=(1, 3))
    # merge
    x = Concatenate(axis = concat_axis)([x_1, x_2, x_3_1, x_3_2, x_4_1, x_4_2])
    return x

# reduction module A
def reduction_A_mod(input, concat_axis = 3):
    # sub_module_1
    x_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)
    # sub_module_2
    x_2 = convolution_unit(input, 384, kernel=(3, 3), strides=(2, 2), padding='valid')
    # sub_module_3
    x_3 = convolution_unit(input, 192, kernel=(1, 1))
    x_3 = convolution_unit(x_3, 224)
    x_3 = convolution_unit(x_3, 256, strides=(2, 2), padding='valid')
    # merge
    x = Concatenate(axis = concat_axis)([x_1, x_2, x_3])
    return x

# reduction module B
def reduction_B_mod(input, concat_axis = 3):
    # sub_module_1
    x_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)
    # sub_module_2
    x_2 = convolution_unit(input, 192, kernel=(1, 1))
    x_2 = convolution_unit(x_2, 192, strides=(2, 2), padding='valid')
    # sub_module_3
    x_3 = convolution_unit(input, 256, kernel=(1, 1))
    x_3 = convolution_unit(x_3, 256, kernel=(1, 7))
    x_3 = convolution_unit(x_3, 320, kernel=(7, 1))
    x_3 = convolution_unit(x_3, 320, strides=(2, 2), padding='valid')
    # merge
    x = Concatenate(axis = concat_axis)([x_1, x_2, x_3])
    return x



def InceptionV4(num_class, image_row, image_column, image_channel=3, learning_rate = 0.045, decay=0.9, epsilon=1.0):
    # create input tensor obj
    image_input = Input(shape=(image_row, image_column, image_channel))
    # stem module
    x = stem(image_input)
    # 4 inception-A
    for a in range(4):
        x = inception_A_mod(x)

    # Reduction-A
    x = reduction_A_mod(x)

    # 7 inception-B
    for b in range(7):
        x = inception_B_mod(x)

    # Reduction-B
    x = reduction_B_mod(x)

    # 3 inception-c
    for c in range(3):
        x = inception_C_mod(x)

    # average pooling layer
    x = AveragePooling2D((8, 8))(x)
    
    # dropout layer with fix rate 0.8
    x = Dropout(0.8)(x)
    x = Flatten()(x)
    # Dense layer with softmax
    out = Dense(num_class, activation = 'softmax')(x)
    # create model
    model = Model(inputs = image_input, outputs = out)
    #model.summary()
    # we use RMSprop as optimizer as in the paper
    rmsprop = RMSprop(learning_rate=learning_rate, epsilon=epsilon, decay=decay)
    # we also try SGD, it does not work better than RMSprop, but you can try
    #sgd = SGD(learning_rate=learning_rate, decay=1e-9, momentum=0.9, nesterov=True)
    # now we compile with crossentropy as loss function
    model.compile(optimizer = rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# doing preprocessing and train the model in this function
def train_models(train_datapath, num_of_classes, log_path, pretrained_weights=None):

    # data list
    Image_list = []
    # label list
    label_list = []

    # preprocessing variables
    image_row, image_column = 299, 299 
    
    # load labels
    Image_label = json.load(open(train_datapath + '//' + 'emotion_labels.json')) # change later with general labels

    # load images with path in label
    for key in Image_label:
        try:
            curr_image = Image.open(train_datapath + '//' + key)
            curr_image = curr_image.resize((image_row, image_column), resample=2) # resize images with bilinear
            curr_image = np.array(curr_image, dtype = np.uint8)
            Image_list.append(curr_image)
            # add labels to the list
            label_list.append(Image_label[key])
        except:
            print("Please check the path of loaded images!")

    # label preprocesss
        # TODO: split the labels with their num positions!

    Images_np = np.array(Image_list)
    Labels_np = np.array(label_list)

    #print(type(label_list), Images_np.shape, Labels_np.shape)


    # pre processing images with minmax
    min_max = MinMaxScaler()
    RGB_pixels = Images_np.reshape(-1, 3)
    transform = min_max.fit_transform(RGB_pixels)
    Image_list_minmax = transform.reshape(Images_np.shape)

    # pre processing with mean substraction
    mean_R = np.mean(Images_np[:][:][:][0]).astype(np.int16)
    mean_G = np.mean(Images_np[:][:][:][1]).astype(np.int16)
    mean_B = np.mean(Images_np[:][:][:][2]).astype(np.int16)

    # mean subtract image list
    Image_list_mean_sub = np.array([(Images_np[i] - np.array([mean_R, mean_G, mean_B])) for i in range(len(Image_list))])

    # now we can create the dataset with sci-kit learn
    X_train, X_val, y_train, y_val = train_test_split(Image_list_mean_sub, Labels_np, test_size=0.25, random_state=154)

    # turn label to one hot encoding
    y_train = to_categorical(y_train, num_of_classes)
    y_val = to_categorical(y_val, num_of_classes)

    batch_size = 20
    Epochs = 100
    # we only use InceptionV4 in here
    model = InceptionV4(num_of_classes, image_row, image_column)

    # load pretrained weights if available
    if pretrained_weights != None:
        model.load_weights(pretrained_weights)

    # now we fit the model and see how training goes
    train_history = model.fit(X_train, y_train, batch_size=batch_size, epochs=Epochs, shuffle=True, verbose=1, validation_data=(X_val, y_val), callbacks=[LrChanging()])

    # create log dir if not exist
    if os.path.isdir(log_path) == False:
        os.mkdir(log_path)

    # Give model file a unique name
    model_name = "{uuid}_{date:%Y%m%d}_".format(uuid = uuid.uuid4(), date = datetime.now()) + "InceptionV4.keras"

    # we want to save the training weight
    model.save_weights(log_path+"//"+ model_name)

    # now we want to plot training loss and validation loss by using history
    plt.figure(0)
    plt.plot(train_history.history['loss'], label='loss')
    plt.plot(train_history.history['val_loss'], label='val_loss')
    plt.title("Training Loss graph")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend(loc = 'upper right')
    plt.show()

    plt.figure(1)
    plt.plot(train_history.history['accuracy'], label='accuracy')
    plt.plot(train_history.history['val_accuracy'], label='val_accuracy')
    plt.title("Training Accuracy graph")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc = 'upper right')
    plt.show()

    return model_name
    
# This one just for detect an image with face in it, use print to print result so far
def detect_image(input_image, weight_path, num_of_classes):

    image_row, image_column = 299, 299 
    curr_image = Image.open(input_image)
    curr_image = curr_image.resize((image_row, image_column), resample=2) # resize images with bilinear
    curr_image = np.array(curr_image)

    # pre processing images with minmax
    '''
    min_max = MinMaxScaler()
    RGB_pixels = curr_image.reshape(-1, 3)
    transform = min_max.fit_transform(RGB_pixels)
    curr_image = transform.reshape(curr_image.shape)
    '''

    model = InceptionV4(num_of_classes, image_row, image_column)
    
    model.load_weights(weight_path)
    curr_image = curr_image.reshape(-1, image_row, image_column, 3)

    print(np.argmax(model.predict(curr_image),axis=1))


# for detection: python InceptionV4.py detect --image=./abc.jpg --weights=./logs/fbc32aaf-3193-420b-92f9-ba028cff4454_20231109_InceptionV4.keras 
# for training: python InceptionV4.py train 
if __name__ == '__main__':
    # Single classication problem
    Num_classes = 3
    # Test to train 18 classes together with one Resnet
    Total_classes = 18
    # Get current path
    curr_path = os.getcwd()

    parser = argparse.ArgumentParser(prog='InceptionV4', description='Train and run InceptionV4 for classify animefaces')
    parser.add_argument('action', help="'train' or 'detect'")
    parser.add_argument('--image', help="the image use for classify faces", required=False, metavar="/path/to/anime/face")
    parser.add_argument('--log', help="path to store checkpoints", required=False,
                        default=os.path.join(curr_path, "logs"), metavar="/path/to/log/folder")
    parser.add_argument('--weights', required=False, default=None, metavar="/path/to/weights.keras",help="*.h5/*.keras files")
    parser.add_argument('--source', required=False, default=curr_path, metavar="/path/to/training/file",help="image and label files")
    

    args = parser.parse_args()

    # the default file path, use --source to modify it
    default_datapath = os.path.join(curr_path, "animeFaces")

    if args.action == "train":
            if args.weights != None:
                try:
                    os.path.isfile(args.weights)
                    print("Load weights from: "+args.weights)
                    model_name = train_models(default_datapath, Num_classes, args.log, pretrained_weights = args.weights)
                    print(args.log + "//"+ model_name)
                except Exception as msg:
                    print(msg)
                    print("Please input a valid weight file! Or check the path of file")
            elif args.source != curr_path:
                print("loading image from source: "+args.source)
                model_name = train_models(args.source, Num_classes, args.log)
                print(args.log + "//"+ model_name)
            else:
                print("loading image from source: "+curr_path+"\\animeFaces")
                model_name = train_models(default_datapath, Num_classes, args.log)
                # should print the name of the weight file
                print(args.log + "//"+ model_name)
    
    elif args.action == "detect":
        if args.weights != None:
            print("Load weights from: "+args.weights)
            detect_image(args.image, args.weights, Num_classes)
        else:
            print("Weight is required for detect image")
        pass
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.action))


