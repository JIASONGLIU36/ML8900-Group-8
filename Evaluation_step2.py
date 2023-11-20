# Author: Fangdi Liu
# Evaluate Step 2 algorithms: Resnet, VGG, InceptionV4. Confusion matrix and other metrics are
# given
import os
import json
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler

import InceptionV4
import ResNet
import VGG


def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test),axis=1)
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

def load_dataset(model_name, model_type=None):
    num_of_classes = 2

    Image_list = []
    label_list = []

    curr_path = os.getcwd()
    data_path = os.path.join(curr_path, "Test_set_step2")
    filenames = os.listdir(data_path)

    if model_name == "InceptionV4":
        image_row, image_column = 299, 299
        model = InceptionV4.InceptionV4(num_of_classes, image_row, image_column)
    elif model_name == "ResNet":
        image_row, image_column = 224, 224
        if model_type == "resnet34":
            model = ResNet.resnet34(num_of_classes)
        elif model_type == "resnet50":
            model = ResNet.resnet50(num_of_classes)
        elif model_type == "resnet101":
            model = ResNet.resnet101(num_of_classes)
        else:
            print("Input model error.")
    elif model_name == "VGG":
        image_row, image_column = 224, 224
        if model_type == "VGG16":
            model = VGG.VGG16(num_of_classes, image_row, image_column, False)
        elif model_type == "VGG19":
            model = VGG.VGG19(num_of_classes, image_row, image_column, False)
        else:
            print("Input model error.")
    else:
        print("Input model error.")

    for each in filenames:
        if each[-5:] == '.json':
            file = open(data_path+ '\\' + each)
            curr_label = json.load(file)
            curr_image = Image.open(data_path + '//' + curr_label['imagePath'])
            curr_image = curr_image.resize((image_row, image_column), resample=2)
            curr_image = np.array(curr_image)
            curr_image = curr_image.reshape(-1, image_row, image_column, 3)
            curr_label = curr_label['shapes'][0]['label']

            label_list.append(curr_label)
            Image_list.append(curr_image)

    X_test = np.array(Image_list)
    X_test = X_test.reshape((161, 224, 224, 3))
    y_test = np.array(label_list)
    y_test = y_test.astype(int)

    return model, X_test, y_test


# python Evaluation_step2.py ResNet --model_type resnet50 --weights=./logs/7f9288b3-3ba8-4afd-9220-4d6bdf14f603_20231116_resnet50.keras
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Evaluation_step2')
    parser.add_argument('model_name', help="Choose the model 'ResNet', 'VGG', or 'InceptionV4' Then choose a specific model with --model_type (except InceptionV4)")
    parser.add_argument('--model_type', required=False, default=None, help="For ResNet, choose 'resnet34', 'resnet50' or 'resnet101'; for VGG, choose 'VGG16' or 'VGG19'")
    parser.add_argument('--weights', required=True, help="Choose a weight file")

    args = parser.parse_args()


    model, X_test, y_test = load_dataset(args.model_name, args.model_type)
    model.load_weights(args.weights)

    evaluate_model(model, X_test, y_test)