# Author: Jiasong Liu
# This is the python file for evaluate models.
# we decided to evaluate 2 steps separately because hardware limit and time constraints
# This file only evaluate Step 1 models
# all test set images sizes are 512*512 
# PS: This program is running very slow without GPU support
import skimage
import json
import numpy as np

import os
import animeface
import animeface_faster

# In evalutation, we send batch of images to locate the anime faces
# we evaluate mask and roi here.
def Evalution_function_step1(image_batch_path, label_path, iou_threshold):

    image_mask_detect_result = {}
    image_bbox_detect_result = {}
    # load label data
    label = json.load(open(label_path))

    # initialize models
    model = animeface.init_model('./logs/Mask_RCNN_animeFace.h5')
    model_Faster = animeface_faster.init_model('./logs/Faster_RCNN_animeFace.h5')

    # initialize True/False positive list (True positive, False positive, False negative)
    Mask_pred_list = [0, 0, 0]
    Faster_pred_list = [0, 0, 0]

    # for every image in test set
    for img in os.listdir(image_batch_path):
        if img[-4:] == '.jpg':
            curr_file = os.path.join(image_batch_path, img)
            print("Current file: " + curr_file)

            image_mask_detect_result[img] = animeface.detection_evalution(model, curr_file)
            image_bbox_detect_result[img] = animeface_faster.detection_evalution(model_Faster, curr_file)
            # now we compare bbox and mask in matrix, we first create mask and bbox from label coordinates

            # initialize mask list for label
            label_mask_x = []
            label_mask_y = []
            current_mask_label = []
            current_bbox_label = []
            bbox_detection = []

            # create label matrix for masks
            for i in range(len(label[img])):
                labeled_matrix = np.zeros((512, 512), dtype=bool)
                label_mask_x = [int(coord[0]) for coord in label[img][i]['Mask']]
                label_mask_y = [int(coord[1]) for coord in label[img][i]['Mask']]
                mask_y, mask_x = skimage.draw.polygon(label_mask_y, label_mask_x)
                labeled_matrix[mask_y, mask_x] = True
                current_mask_label.append(labeled_matrix)

            # create label matrix for bbox
            for i in range(len(label[img])):
                labeled_matrix = np.zeros((512, 512), dtype=bool)
                label_bbox_start = [int(label[img][i]['bbox'][0][1]), int(label[img][i]['bbox'][0][0])]
                label_bbox_extent = [(int(label[img][i]['bbox'][1][1]) - int(label[img][i]['bbox'][0][1])), (int(label[img][i]['bbox'][1][0]) - int(label[img][i]['bbox'][0][0]))]
                bbox_y, bbox_x = skimage.draw.rectangle(label_bbox_start, extent=label_bbox_extent, shape=labeled_matrix.shape)
                labeled_matrix[bbox_y, bbox_x] = True
                current_bbox_label.append(labeled_matrix)

            # create bbox for detection result
            for i in range(image_bbox_detect_result[img].shape[0]):
                detect_matrix = np.zeros((512, 512), dtype=bool)
                detect_bbox_start = [image_bbox_detect_result[img][i][0], image_bbox_detect_result[img][i][1]]
                detect_bbox_extent = [(image_bbox_detect_result[img][i][2] - image_bbox_detect_result[img][i][0]), (image_bbox_detect_result[img][i][3] - image_bbox_detect_result[img][i][1])]
                bbox_y, bbox_x = skimage.draw.rectangle(detect_bbox_start, extent=detect_bbox_extent, shape=detect_matrix.shape)
                detect_matrix[bbox_y, bbox_x] = True
                bbox_detection.append(detect_matrix)


            # Compute Mask RCNN IoU and count True/False positive and False negative
            # initial a list for check if the prediction is correct
            label_count_check = [False for _ in range(len(current_mask_label))] 
            mask_count_check = [False for _ in range(image_mask_detect_result[img].shape[2])]
            # initial number of prediction result
            False_positive = 0
            False_negative = 0
            True_positive = 0
            for i in range(image_mask_detect_result[img].shape[2]):
                for j in range(len(current_mask_label)):
                    # now we compare the areas by compute IOU by threshold
                    # use OR operation to get union areas
                    Union = [[mask_ele or mask_label_ele for mask_ele, mask_label_ele in zip(mask_row, mask_label_row)] for mask_row, mask_label_row in zip(image_mask_detect_result[img][:, :, i], current_mask_label[j])]
                    # use AND operation to get intersect areas
                    Intersect = [[mask_ele and mask_label_ele for mask_ele, mask_label_ele in zip(mask_row, mask_label_row)] for mask_row, mask_label_row in zip(image_mask_detect_result[img][:, :, i], current_mask_label[j])]

                    IoU = np.array(Intersect).sum()/np.array(Union).sum()
                    if IoU >= iou_threshold and label_count_check[j] == False:
                        label_count_check[j] = True
                        # for each mask we got True, we don't check again
                        mask_count_check[i] = True
            
            True_positive = label_count_check.count(True)
            Mask_pred_list[0] += True_positive
            False_negative = label_count_check.count(False)
            Mask_pred_list[2] += False_negative
            False_positive = mask_count_check.count(False)
            Mask_pred_list[1] += False_positive

            # local result for 1 image
            print("MaskRCNN: False positive: "+ str(False_positive) + " True positive: " + str(True_positive) + " False negative: " + str(False_negative))

            # now we compute Faster RCNN
            label_count_check = [False for _ in range(len(current_bbox_label))] 
            bbox_count_check = [False for _ in range(len(bbox_detection))]
            # initial number of prediction result
            False_positive = 0
            False_negative = 0
            True_positive = 0
            for i in range(len(bbox_detection)):
                for j in range(len(current_bbox_label)):
                    # now we compare the areas by compute IOU by threshold
                    # use OR operation to get union areas
                    Union = [[bbox_ele or bbox_label_ele for bbox_ele, bbox_label_ele in zip(bbox_row, bbox_label_row)] for bbox_row, bbox_label_row in zip(bbox_detection[i], current_bbox_label[j])]
                    # use AND operation to get intersect areas
                    Intersect = [[bbox_ele and bbox_label_ele for bbox_ele, bbox_label_ele in zip(bbox_row, bbox_label_row)] for bbox_row, bbox_label_row in zip(bbox_detection[i], current_bbox_label[j])]

                    IoU = np.array(Intersect).sum()/np.array(Union).sum()
                    if IoU >= iou_threshold and label_count_check[j] == False:
                        label_count_check[j] = True
                        # for each bbox we got True, we don't check again
                        bbox_count_check[i] = True
            
            True_positive = label_count_check.count(True)
            Faster_pred_list[0] += True_positive
            False_negative = label_count_check.count(False)
            Faster_pred_list[2] += False_negative
            False_positive = bbox_count_check.count(False)
            Faster_pred_list[1] += False_positive

            # local result for 1 image
            print("FasterRCNN: False positive: "+ str(False_positive) + " True positive: " + str(True_positive) + " False negative: " + str(False_negative))

                
    # now we can compute Precision, Recall and F1 score 
    print("Evaluation for Mask RCNN:")         
    precision = Mask_pred_list[0]/(Mask_pred_list[0]+Mask_pred_list[1])
    recall = Mask_pred_list[0]/(Mask_pred_list[0]+Mask_pred_list[2])
    print("Precision: {:.2%}".format(precision))      
    print("Recall: {:.2%}".format(recall))     
    print("F1 score: {:.2%}".format((2*precision*recall)/(precision+recall)))    

    print("Evaluation for Faster RCNN:")         
    precision = Faster_pred_list[0]/(Faster_pred_list[0]+Faster_pred_list[1])
    recall = Faster_pred_list[0]/(Faster_pred_list[0]+Faster_pred_list[2])
    print("Precision: {:.2%}".format(precision))      
    print("Recall: {:.2%}".format(recall))     
    print("F1 score: {:.2%}".format((2*precision*recall)/(precision+recall)))   


if __name__ == '__main__':
    
    # we need to first get all data 
    RCNN_Testset_path = './Test_set_step1'
    label = './Test_set_step1/test_label.json'

    # evaluate mask and faster RCNN, IOU threshold set to 0.5 (since the training data is too low for this model)
    Evalution_function_step1(RCNN_Testset_path, label, 0.5)


    print('End of evalutation')

