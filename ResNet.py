# Author: Jiasong Liu, Fangdi Liu
import os
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dense,\
Flatten, Add, BatchNormalization

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



def identity_mapping_module(input_tensor, filter_List, stage, block, bn_axis = 3, kernel = (3, 3)):
    # 3 different channel number for this identity mapping
    filter1, filter2, filter3 = filter_List
    conv_name = 'res' + str(stage) + block + '_module'
    bn_name = 'bn' + str(stage) + block + '_module'
    # module 2a
    x = Conv2D(filter1, (1, 1), name = conv_name + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name = bn_name+'2a')(x)
    x = Activation('relu')(x)
    # module 2b
    x = Conv2D(filter2, kernel, padding = 'same', name = conv_name + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name = bn_name + '2b')(x)
    x = Activation('relu')(x)
    # module 2c
    x = Conv2D(filter3, (1, 1), name = conv_name + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name = bn_name+'2c')(x)
    # don't need activation function here since we are adding and activate with x + f(x)

    # Now add input and output of the non linear layers together
    x = Add(name=conv_name+'_add')([x, input_tensor])
    x = Activation('relu', name = 'res' + str(stage) + block+'_out')(x)
    return x

def match_dimension_module(input_tensor, filter_List, stage, block, bn_axis = 3, strides = (2, 2), kernel = (3, 3)):
    # using 1*1 conv to convert filter size to perform add operation
    filter1, filter2, filter3 = filter_List
    conv_name = 'res' + str(stage) + block + '_module'
    bn_name = 'bn' + str(stage) + block + '_module'
    # module 2a
    x = Conv2D(filter1, (1, 1), strides = strides, name=conv_name+ '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name = bn_name+ '2a')(x)
    x = Activation('relu')(x)
    # module 2b
    x = Conv2D(filter2, kernel, padding = 'same', name=conv_name+'2b')(x)
    x = BatchNormalization(axis=bn_axis, name = bn_name + '2b')(x)
    x = Activation('relu')(x)
    # module 2c
    x = Conv2D(filter3, (1, 1), name=conv_name+ '2c')(x)
    x = BatchNormalization(axis=bn_axis, name = bn_name+ '2c')(x)
    # don't need activation function here since we are adding and activate with x + f(x)
    # now we doing a 1*1 conv to get same filters
    shortcut = Conv2D(filter3, (1, 1), strides = strides, name = conv_name + '_1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name = bn_name + '_1')(shortcut)

    x = Add(name=conv_name+'_add')([x, shortcut])
    x = Activation('relu', name = 'res' + str(stage) + block+'_out')(x)
    return x


def identity_mapping_module_2layers(input_tensor, filter, stage, block, bn_axis = 3, kernel = (3, 3)):
    conv_name = 'res' + str(stage) + block + '_module'
    bn_name = 'bn' + str(stage) + block + '_module'

    # module 2a
    x = Conv2D(filter, kernel, padding = 'same', name = conv_name + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name = bn_name + '2a')(x)
    x = Activation('relu')(x)
    # module 2b
    x = Conv2D(filter, kernel, padding = 'same', name = conv_name + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name = bn_name+'2b')(x)
    # don't need activation function here since we are adding and activate with x + f(x)

    # Now add input and output of the non linear layers together
    x = Add(name=conv_name+'_add')([x, input_tensor])
    x = Activation('relu', name = 'res' + str(stage) + block+'_out')(x)
    return x

def match_dimension_module_2layers(input_tensor, filter, stage, block, bn_axis = 3, strides = (2, 2), kernel = (3, 3)):
    # using 1*1 conv to convert filter size to perform add operation
    conv_name = 'res' + str(stage) + block + '_module'
    bn_name = 'bn' + str(stage) + block + '_module'
    # module 2a
    x = Conv2D(filter, kernel, padding = 'same', strides = strides, name=conv_name+ '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name = bn_name+ '2a')(x)
    x = Activation('relu')(x)
    # module 2b
    x = Conv2D(filter, kernel, padding = 'same', name=conv_name+'2b')(x)
    x = BatchNormalization(axis=bn_axis, name = bn_name + '2b')(x)
    # don't need activation function here since we are adding and activate with x + f(x)
    # now we doing a 1*1 conv to get same filters
    shortcut = Conv2D(filter, (1, 1), strides = strides, name = conv_name + '_1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name = bn_name + '_1')(shortcut)

    x = Add(name=conv_name+'_add')([x, shortcut])
    x = Activation('relu', name = 'res' + str(stage) + block+'_out')(x)
    return x

# in case the model is too complex, we are also going to try 34 layers 
def resnet34(num_class, bn_axis = 3, image_channel=3, learning_rate = 1e-5, decay=1e-9, momentum=0.9):
    image_row, image_column = 224, 224 # default size from paper
    # create input tensor obj
    image_input = Input(shape=(image_row, image_column, image_channel))
    # Stage 1
    x = ZeroPadding2D((3, 3))(image_input) # pad to h and w to avoid reduction in size
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x) # use batchnormalization on the filters
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # stage 2 3 * [64, 64]
    x = match_dimension_module_2layers(x, 64, stage=2, block='a', strides=(1, 1))
    x = identity_mapping_module_2layers(x, 64, stage=2, block='b')
    x = identity_mapping_module_2layers(x, 64, stage=2, block='c')
    # stage 3 4 * [128, 128]
    x = match_dimension_module_2layers(x, 128, stage=3, block='a')
    x = identity_mapping_module_2layers(x, 128, stage=3, block='b')
    x = identity_mapping_module_2layers(x, 128, stage=3, block='c')
    x = identity_mapping_module_2layers(x, 128, stage=3, block='d')
    # stage 4 6 * [256, 256]
    x = match_dimension_module_2layers(x, 256, stage=4, block='a')
    x = identity_mapping_module_2layers(x, 256, stage=4, block='b')
    x = identity_mapping_module_2layers(x, 256, stage=4, block='c')
    x = identity_mapping_module_2layers(x, 256, stage=4, block='d')
    x = identity_mapping_module_2layers(x, 256, stage=4, block='e')
    x = identity_mapping_module_2layers(x, 256, stage=4, block='f')
    # stage 5 3 * [512, 512]
    x = match_dimension_module_2layers(x, 512, stage=5, block='a')
    x = identity_mapping_module_2layers(x, 512, stage=5, block='b')
    # finally do average pooling and fully connected layer
    x_fully_connect = AveragePooling2D((7, 7), name='average_pooling')(x)
    x_fully_connect = Flatten()(x_fully_connect)
    x_fully_connect = Dense(num_class, activation = 'softmax', name = 'fc'+str(num_class))(x_fully_connect)
    # now create a model
    model = Model(inputs = image_input, outputs = x_fully_connect)
    # use SGD as specified in paper
    sgd = SGD(learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    # now we compile with crossentropy as loss function
    model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# The most customize way to implement a resnet model
def resnet50(num_class, bn_axis = 3, image_channel=3, learning_rate = 1e-5, decay=1e-9, momentum=0.9):
    image_row, image_column = 224, 224 # default size from paper
    # create input tensor obj
    image_input = Input(shape=(image_row, image_column, image_channel))

    # Stage 1
    x = ZeroPadding2D((3, 3))(image_input) # pad to h and w to avoid reduction in size
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x) # use batchnormalization on the filters
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # Stage 2: 3 * [64, 64, 256]
    x = match_dimension_module(x, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_mapping_module(x, [64, 64, 256], stage=2, block='b')
    x = identity_mapping_module(x, [64, 64, 256], stage=2, block='c')
    # Stage 3: 4 * [128, 128, 512]
    x = match_dimension_module(x, [128, 128, 512], stage=3, block='a')
    x = identity_mapping_module(x, [128, 128, 512], stage=3, block='b')
    x = identity_mapping_module(x, [128, 128, 512], stage=3, block='c')
    x = identity_mapping_module(x, [128, 128, 512], stage=3, block='d')
    # Stage 4: 6 * [256, 256, 1024]
    x = match_dimension_module(x, [256, 256, 1024], stage=4, block='a')
    x = identity_mapping_module(x, [256, 256, 1024], stage=4, block='b')
    x = identity_mapping_module(x, [256, 256, 1024], stage=4, block='c')
    x = identity_mapping_module(x, [256, 256, 1024], stage=4, block='d')
    x = identity_mapping_module(x, [256, 256, 1024], stage=4, block='e')
    x = identity_mapping_module(x, [256, 256, 1024], stage=4, block='f')
    # Stage 5: 3 * [512, 512, 2048]
    x = match_dimension_module(x, [512, 512, 2048], stage=5, block='a')
    x = identity_mapping_module(x, [512, 512, 2048], stage=5, block='b')
    x = identity_mapping_module(x, [512, 512, 2048], stage=5, block='c')
    # finally do average pooling and fully connected layer
    x_fully_connect = AveragePooling2D((7, 7), name='average_pooling')(x)
    x_fully_connect = Flatten()(x_fully_connect)
    x_fully_connect = Dense(num_class, activation = 'softmax', name = 'fc'+str(num_class))(x_fully_connect)
    # now create a model
    model = Model(inputs = image_input, outputs = x_fully_connect)
    # use SGD as specified in paper
    sgd = SGD(learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    # now we compile with crossentropy as loss function
    model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# resnet101 is almost same with resnet 50, the only difference is in stage 4
def resnet101(num_class, bn_axis = 3, image_channel=3, learning_rate = 1e-5, decay=1e-9, momentum=0.9):
    image_row, image_column = 224, 224 # default size from paper
    # create input tensor obj
    image_input = Input(shape=(image_row, image_column, image_channel))
    # Stage 1
    x = ZeroPadding2D((3, 3))(image_input) # pad to h and w to avoid reduction in size
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x) # use batchnormalization on the filters
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # Stage 2: 3 * [64, 64, 256]
    x = match_dimension_module(x, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_mapping_module(x, [64, 64, 256], stage=2, block='b')
    x = identity_mapping_module(x, [64, 64, 256], stage=2, block='c')
    # Stage 3: 4 * [128, 128, 512]
    x = match_dimension_module(x, [128, 128, 512], stage=3, block='a')
    x = identity_mapping_module(x, [128, 128, 512], stage=3, block='b')
    x = identity_mapping_module(x, [128, 128, 512], stage=3, block='c')
    x = identity_mapping_module(x, [128, 128, 512], stage=3, block='d')
    # Stage 4: 23 * [256, 256, 1024]
    x = match_dimension_module(x, [256, 256, 1024], stage=4, block='_4-1')
    # 23 - 1 modules
    for i in range(1, 23):
        block_label = '_4-' + str(i+1)
        x = identity_mapping_module(x, [256, 256, 1024], stage=4, block=block_label)
    # Stage 5: 3 * [512, 512, 2048]
    x = match_dimension_module(x, [512, 512, 2048], stage=5, block='a')
    x = identity_mapping_module(x, [512, 512, 2048], stage=5, block='b')
    x = identity_mapping_module(x, [512, 512, 2048], stage=5, block='c')
    # finally do average pooling and fully connected layer
    x_fully_connect = AveragePooling2D((7, 7), name='average_pooling')(x)
    x_fully_connect = Flatten()(x_fully_connect)
    x_fully_connect = Dense(num_class, activation = 'softmax', name = 'fc'+str(num_class))(x_fully_connect)
    # now create a model
    model = Model(inputs = image_input, outputs = x_fully_connect)
    # use SGD as specified in paper
    sgd = SGD(learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    # now we compile with crossentropy as loss function
    model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# doing preprocessing and train the model in this function
def train_models(train_datapath, feature, model_type, log_path, pretrained_weights=None):

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

    #print(type(label_list), Images_np.shape, Labels_np.shape)


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
    if model_type == 'resnet34':
        model = resnet34(num_class = num_of_classes)
    elif model_type == 'resnet50':
        model = resnet50(num_class = num_of_classes)
    elif model_type == 'resnet101':
        model = resnet101(num_class = num_of_classes)
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
def detect_image(input_image, model_type, weight_path, feature):

    num_of_classes = Total_classes[feature]

    image_row, image_column = 224, 224 
    curr_image = Image.open(input_image)
    curr_image = curr_image.resize((image_row, image_column), resample=2) # resize images with bilinear
    curr_image = np.array(curr_image)


    if model_type == 'resnet34':
        model = resnet34(num_class = num_of_classes)
    elif model_type == 'resnet50':
        model = resnet50(num_class = num_of_classes)
    elif model_type == 'resnet101':
        model = resnet101(num_class = num_of_classes)
    else:
        print('Unexpected error!' + model_type)
        exit()
    
    model.load_weights(weight_path)
    curr_image = curr_image.reshape(-1, image_row, image_column, 3)

    print(np.argmax(model.predict(curr_image),axis=1))
    return (np.argmax(model.predict(curr_image),axis=1))[0].astype(int)



# for detection: python Resnet.py detect hairstyle --image=./abc.jpg --weights=./logs/7c98ff7f-7fa5-4557-840d-3d2952af1790_20231108_resnet34.keras --model=resnet34
# for training: python Resnet.py train hairstyle --model resnet34 --weights=./logs/b1198048-e1a1-4fe4-8f8d-1953d093b2db_20231108_resnet34.keras
if __name__ == '__main__':
    # Single classication problem
    # Num_classes = 3
    # Test to train 18 classes together with one Resnet
    # Total_classes = 18
    # Get current path
    curr_path = os.getcwd()

    parser = argparse.ArgumentParser(prog='ResNet', description='Train and run different Resnet for classify animefaces')
    parser.add_argument('action', help="'train' or 'detect'")
    parser.add_argument('feature', help="'hairstyle' or 'emotion' or 'eyecolor'")
    parser.add_argument('--image', help="the image use for classify faces", required=False, metavar="/path/to/anime/face")
    parser.add_argument('--log', help="path to store checkpoints", required=False,
                        default=os.path.join(curr_path, "logs"), metavar="/path/to/resnet/log/folder")
    parser.add_argument('--weights', required=False, default=None, metavar="/path/to/weights.keras",help="*.h5/*.keras files")
    parser.add_argument('--source', required=False, default=curr_path, metavar="/path/to/training/file",help="image and label files")
    parser.add_argument('--model', required=True, choices=['resnet34', 'resnet50', 'resnet101'], help="Choose what model to train or detect \
                        (model and weight should match)")
    

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
                model_name = train_models(args.source, args.feature, args.model, args.log)
                print(args.log + "//"+ model_name)
            else:
                print("loading image from source: "+curr_path+"\\animeFaces")
                model_name = train_models(default_datapath, args.feature, args.model, args.log)
                # should print the name of the weight file
                print(args.log + "//"+ model_name)
    
    elif args.action == "detect":
        if args.weights != None:
            print("Load weights from: "+args.weights)
            detect_image(args.image, args.model, args.weights, args.feature)
        else:
            print("Weight is required for detect image")
        pass
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.action))


