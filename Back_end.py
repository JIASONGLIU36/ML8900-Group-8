# Author: Jiasong Liu
# This is the back_end for the model
from PIL import Image, ImageFont, ImageDraw

import animeface
import animeface_faster
import ResNet
import VGG
import InceptionV4

# all possible output condition, if some of the model failed to run, it will return false
All_output_conditions = {'MaskRCNN':False, 'Resnet34_emotion':False, 'Resnet34_hairstyle':False, 'Resnet34_eyecolor':False,
                         'Resnet50_emotion':False, 'Resnet50_hairstyle':False, 'Resnet50_eyecolor':False,
                         'Resnet101_emotion':False, 'Resnet101_hairstyle':False, 'Resnet101_eyecolor':False,
                         'VGG16_emotion':False, 'VGG16_hairstyle':False, 'VGG16_eyecolor':False,
                         'VGG19_emotion':False, 'VGG19_hairstyle':False, 'VGG19_eyecolor':False,
                         'InceptionV4_emotion':False, 'InceptionV4_hairstyle':False, 'InceptionV4_eyecolor':False,
                         }


def Resnet(crop_image, feature, model):
    try:
        result = ResNet.detect_image(crop_image, 'resnet'+model, './logs/' + 'resnet'+model+'_'+feature+".keras", feature=feature)
        # running without error
        All_output_conditions['Resnet'+model+'_'+feature] = True
        return result
    except Exception as e:
        # create a file with error
        All_output_conditions['Resnet'+model+'_'+feature] = False
        print(e)
        return -1
    

def VGG_model(crop_image, feature, model):
    try:
        result = VGG.detect_image(crop_image, 'VGG'+model, './logs/' + 'VGG'+model+'_'+feature+".keras", feature, False)
        # running without error
        All_output_conditions['VGG'+model+'_'+feature] = True
        return result
    except Exception as e:
        # create a file with error
        All_output_conditions['VGG'+model+'_'+feature] = False
        print(e)
        return -1
    
def InceptionV4_model(crop_image, feature):
    try:
        result = InceptionV4.detect_image(crop_image, './logs/' + 'InceptionV4_'+feature+".keras", feature=feature)
        # running without error
        All_output_conditions['InceptionV4_'+feature] = True
        return result
    except Exception as e:
        # create a file with error
        All_output_conditions['InceptionV4_'+feature] = False
        print(e)
        return -1


def MaskRCNN_FasterRCNN(image, image_id):
    try:
        model = animeface.init_model('./logs/Mask_RCNN_animeFace.h5')
        model_Faster = animeface_faster.init_model('./logs/Faster_RCNN_animeFace.h5')
        image_location = animeface.color_filter(model, image, image_id=image_id)
        image_location_Faster = animeface_faster.color_filter(model_Faster, image, image_id=image_id)
        # running without error
        All_output_conditions['MaskRCNN'] = True
        return [image_location, image_location_Faster]
    except Exception as e:
        All_output_conditions['MaskRCNN'] = False
        print(e)
        return [[[], []], [[], []]]  # assign null for unsuccessful run 
    
# store the roi results
Detection_result = []

def DetectionProcess(image, image_id):
    # initialize method
    result_mask_rcnn = MaskRCNN_FasterRCNN(image, image_id)

    print(result_mask_rcnn[0][1])
    Faster_pic = Image.open('./temp_img/'+image_id+'_faster.png')
    Mask_pic = Image.open('./temp_img/'+image_id+'_mask.png')
    # if we found more than 1 faces then we label the faces and running the step 2 models
    # else we just pass the origin image as result
    if len(result_mask_rcnn[0][0]) > 0:
        draw_Mask = ImageDraw.Draw(Mask_pic)
        # first we draw numbers on the image
        for key, i in enumerate(result_mask_rcnn[0][0]):
            draw_Mask.text((i[1], i[0]), str(key), (255,0,0), font=ImageFont.truetype('arial.ttf', 20))
        # then we save the 2 images for output
        Mask_pic.save('./temp_img/'+image_id+'_mask.png')
    else:
        pass
    
    # now we draw number for Faster RCNN
    if len(result_mask_rcnn[1][0]) > 0:
        draw_Faster = ImageDraw.Draw(Faster_pic)
        # first we draw numbers on the image
        for key, i in enumerate(result_mask_rcnn[1][0]):
            draw_Faster.text((i[1], i[0]), str(key), (255,0,0), font=ImageFont.truetype('arial.ttf', 20))
        # then we save the 2 images for output
        Faster_pic.save('./temp_img/'+image_id+'_faster.png')

    return [len(result_mask_rcnn[0][0]), len(result_mask_rcnn[1][0]), All_output_conditions]

def step_2_models_Faster(image_id, crop_face_id, feature, model = ''):
    # temp images folder
    image_id = './temp_img/' + image_id
    
    #Detection_result for Faster RCNN
    if model == 'Resnet 34':
        Faster_result = Resnet(image_id+'_VGG_RES_FASTER_'+str(crop_face_id)+'.png', feature, '34')
    elif model == 'Resnet 50':
        Faster_result = Resnet(image_id+'_VGG_RES_FASTER_'+str(crop_face_id)+'.png', feature, '50')
    elif model == 'Resnet 101':
        Faster_result = Resnet(image_id+'_VGG_RES_FASTER_'+str(crop_face_id)+'.png', feature, '101')
    elif model == 'VGG 16':
        Faster_result = VGG_model(image_id+'_VGG_RES_FASTER_'+str(crop_face_id)+'.png', feature, '16')
    elif model == 'VGG 19':
        Faster_result = VGG_model(image_id+'_VGG_RES_FASTER_'+str(crop_face_id)+'.png', feature, '19')
    elif model == 'Inception V4':
        Faster_result = InceptionV4_model(image_id+'_V4_FASTER_'+str(crop_face_id)+'.png', feature)
    else:
        raise Exception("Unknown model selected! Please check AnimeFaceDetection.py for debug") 

    return Faster_result

def step_2_models_Mask(image_id, crop_face_id, feature, model = ''):
    # temp images folder
    image_id = './temp_img/' + image_id

    #Detection_result for Mask RCNN
    if model == 'Resnet 34':
        Mask_result = Resnet(image_id+'_VGG_RES_MASK_'+str(crop_face_id)+'.png', feature, '34')
    elif model == 'Resnet 50':
        Mask_result = Resnet(image_id+'_VGG_RES_MASK_'+str(crop_face_id)+'.png', feature, '50')
    elif model == 'Resnet 101':
        Mask_result = Resnet(image_id+'_VGG_RES_MASK_'+str(crop_face_id)+'.png', feature, '101')
    elif model == 'VGG 16':
        Mask_result = VGG_model(image_id+'_VGG_RES_MASK_'+str(crop_face_id)+'.png', feature, '16')
    elif model == 'VGG 19':
        Mask_result = VGG_model(image_id+'_VGG_RES_MASK_'+str(crop_face_id)+'.png', feature, '19')
    elif model == 'Inception V4':
        Mask_result = InceptionV4_model(image_id+'_V4_MASK_'+str(crop_face_id)+'.png', feature)
    else:
        raise Exception("Unknown model selected! Please check AnimeFaceDetection.py for debug") 

    return Mask_result

print("Backend Initialization Complete")