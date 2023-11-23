# Author: Jiasong Liu, Fangdi Liu
# This is implementation for diffenent VGG models, which are VGG 16, 19.
import os
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from sklearn.model_selection import train_test_split

import numpy as np


Total_classes = {'hairstyle': 2, 'emotion': 2, 'eyecolor': 9}

class LrChanging(tf.keras.callbacks.Callback):
    # doing a little control about loss here, if the loss presist, we divide learning rate by 10
    global Last_loss, high_loss_count
    Last_loss = 9999
    high_loss_count = 0

    def on_epoch_end(self, epoch, logs=None):
        curr_loss = logs['loss']
        lr = self.model.optimizer.lr.read_value()
        global Last_loss, high_loss_count
        if epoch > 0:
            if high_loss_count == 8:
                self.model.optimizer.lr.assign(lr/10)
                high_loss_count = 0
    
        if Last_loss <= curr_loss:
            high_loss_count+=1
        else:
            Last_loss = curr_loss # update loss
        print("Current lr:" + str(lr.numpy()))

# Configuration D, with training parameters in the origin paper
def VGG16(num_of_classes, image_row, image_column, enable_dropout, image_channel=3, learning_rate = 1e-2, decay=5e-4, momentum=0.9):
    # create input tensor obj
    image_input = Input(shape=(image_row, image_column, image_channel))
    # block 1 3*3 64
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv_1')(image_input)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_maxPool')(x)
    # block 2 3*3 128
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_maxPool')(x)
    # block 3 3*3 256
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv_2')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv_3')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_maxPool')(x)
    # block 4 3*3 512
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv_2')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv_3')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_maxPool')(x)
    # block 5 3*3 512
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv_2')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv_3')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_maxPool')(x)    
    # now we use fully connected layer for output
    fc = Flatten(name = 'flatten_layer')(x)
    if enable_dropout == True:
        fc = Dense(4096, activation='relu', name='FC_4096_1_dropout')(fc)
        fc = Dropout(0.5)(fc)
        fc = Dense(4096, activation='relu', name='FC_4096_2_dropout')(fc)
        fc = Dropout(0.5)(fc)
    else:
        fc = Dense(4096, activation='relu', name='FC_4096_1')(fc)
        fc = Dense(4096, activation='relu', name='FC_4096_2')(fc)
    out = Dense(num_of_classes, activation='softmax', name='Output_classification')(fc)

    # now create a model
    model = Model(inputs = image_input, outputs = out)
    # use SGD as specified in paper
    sgd = SGD(learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    # now we compile with crossentropy as loss function
    model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Configuration E
def VGG19(num_of_classes, image_row, image_column, enable_dropout, image_channel=3, learning_rate = 1e-2, decay=5e-4, momentum=0.9):
    # create input tensor obj
    image_input = Input(shape=(image_row, image_column, image_channel))
    # block 1 3*3 64
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv_1')(image_input)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_maxPool')(x)
    # block 2 3*3 128
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_maxPool')(x)
    # block 3 3*3 256 (extra layer for 3*3 conv 256)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv_2')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv_3')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv_4')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_maxPool')(x)
    # block 4 3*3 512 (extra layer for 3*3 conv 512)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv_2')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv_3')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv_4')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_maxPool')(x)
    # block 5 3*3 512 (extra layer for 3*3 conv 512)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv_2')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv_3')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv_4')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_maxPool')(x)    
    # now we use fully connected layer for output
    fc = Flatten(name = 'flatten_layer')(x)
    if enable_dropout == True:
        fc = Dense(4096, activation='relu', name='FC_4096_1_dropout')(fc)
        fc = Dropout(0.5)(fc)
        fc = Dense(4096, activation='relu', name='FC_4096_2_dropout')(fc)
        fc = Dropout(0.5)(fc)
    else:
        fc = Dense(4096, activation='relu', name='FC_4096_1')(fc)
        fc = Dense(4096, activation='relu', name='FC_4096_2')(fc)
    out = Dense(num_of_classes, activation='softmax', name='Output_classification')(fc)

    # now create a model
    model = Model(inputs = image_input, outputs = out)
    # use SGD as specified in paper
    sgd = SGD(learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    # now we compile with crossentropy as loss function
    model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# doing preprocessing and train the model in this function
def train_models(train_datapath, feature, model_type, log_path, enable_dropout, pretrained_weights=None):

    num_of_classes = Total_classes[feature]

    # data list
    Image_list = []
    # label list
    label_list = []

    # preprocessing variables
    image_row, image_column = 224, 224 
    
    # load labels
    Image_label = json.load(open(train_datapath + '//' + feature + '_label.json')) # change later with general labels

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

    print(type(label_list), Images_np.shape, Labels_np.shape)


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

    batch_size = 10
    Epochs = 100
    if model_type == 'VGG16':
        model = VGG16(num_of_classes, image_row, image_column, enable_dropout)
    elif model_type == 'VGG19':
        model = VGG19(num_of_classes, image_row, image_column, enable_dropout)
        model.summary()
    else:
        print('Unexpected error!' + model_type)
        exit()
    # load pretrained weights if available
    if pretrained_weights != None:
        model.load_weights(pretrained_weights)

    # now we fit the model and see how training goes
    train_history = model.fit(X_train, y_train, batch_size=batch_size, epochs=Epochs, shuffle=True, verbose=1, validation_data=(X_val, y_val), callbacks=[LrChanging()])

    # create log dir if not exist
    if os.path.isdir(log_path) == False:
        os.mkdir(log_path)

    # Give model file a constant name (for front end)
    model_name = "{weight_name}_{featureName}".format(weight_name = model_type, featureName = feature) + ".keras"

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
def detect_image(input_image, model_type, weight_path, feature, enable_dropout):

    num_of_classes = Total_classes[feature]

    image_row, image_column = 224, 224 
    curr_image = Image.open(input_image)
    curr_image = curr_image.resize((image_row, image_column), resample=2) # resize images with bilinear
    curr_image = np.array(curr_image)


    if model_type == 'VGG16':
        model = VGG16(num_of_classes, image_row, image_column, enable_dropout)
    elif model_type == 'VGG19':
        model = VGG19(num_of_classes, image_row, image_column, enable_dropout)
    else:
        print('Unexpected error!' + model_type)
        exit()
    
    model.load_weights(weight_path)
    curr_image = curr_image.reshape(-1, image_row, image_column, 3)

    print(np.argmax(model.predict(curr_image),axis=1))
    return (np.argmax(model.predict(curr_image),axis=1))[0].astype(int)


# for detection: python VGG.py detect hairstyle --weights=./logs/1e13f792-edc1-4f8d-be20-7ef99fde40fe_20231108_VGG19.keras --image=./abc.jpg --model VGG19
# for training: python VGG.py train hairstyle --model VGG19 --dropout
if __name__ == '__main__':
    # Single classication problem
    # Num_classes = 3
    # Test to train 18 classes together with one Resnet
    # Total_classes = 18
    # Get current path
    curr_path = os.getcwd()

    parser = argparse.ArgumentParser(prog='VGG', description='Train and run different VGG for classify animefaces')
    parser.add_argument('action', help="'train' or 'detect'")
    parser.add_argument('feature', help="'hairstyle' or 'emotion' or 'eyecolor'")
    parser.add_argument('--image', help="the image use for classify faces", required=False, metavar="/path/to/anime/face")
    parser.add_argument('--log', help="path to store checkpoints", required=False,
                        default=os.path.join(curr_path, "logs"), metavar="/path/to/VGG/log/folder")
    parser.add_argument('--weights', required=False, default=None, metavar="/path/to/weights.keras",help="*.h5/*.keras files")
    parser.add_argument('--source', required=False, default=curr_path, metavar="/path/to/training/file",help="image and label files")
    parser.add_argument('--model', required=True, choices=['VGG16', 'VGG19'], help="Choose what model to train or detect (model and weight should match)")
    parser.add_argument('--dropout', required=False, default=False, action='store_true', help="Choose to use dropout layer or not")
    

    args = parser.parse_args()

    # the default file path, use --source to modify it
    default_datapath = os.path.join(curr_path, "animeFaces")

    if args.action == "train":
            if args.weights != None:
                try:
                    os.path.isfile(args.weights)
                    print("Load weights from: "+args.weights)
                    model_name = train_models(default_datapath, args.feature, args.model, args.log, pretrained_weights = args.weights)
                    print(args.log + "//"+ model_name)
                except Exception as msg:
                    print(msg)
                    print("Please input a valid weight file! Or check the path of file")
            elif args.source != curr_path:
                print("loading image from source: "+args.source)
                model_name = train_models(args.source, args.feature, args.model, args.log, args.dropout)
                print(args.log + "//"+ model_name)
            else:
                print("loading image from source: "+curr_path+"\\animeFaces")
                model_name = train_models(default_datapath, args.feature, args.model, args.log, args.dropout)
                # should print the name of the weight file
                print(args.log + "//"+ model_name)
    
    elif args.action == "detect":
        if args.weights != None:
            print("Load weights from: "+args.weights)
            detect_image(args.image, args.model, args.weights, args.feature, args.dropout)
        else:
            print("Weight is required for detect image")
        pass
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.action))


