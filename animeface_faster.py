# Author: Jiasong Liu
# This class is use mask RCNN to train the faster RCNN. The result for mask is unused
# This works because mask RCNN is training the faster RCNN with bounding box converted from mask
# Therefore, if we feed bounding box, then only faster RCNN part of the Mask RCNN is trained.
import os
import json
import uuid
import utils
import config
import argparse
import skimage
import numpy as np
import model_Faster as ml



# first, let us create label that fit the model
ROOT_PATH = os.getcwd()
LOGDIR = os.path.join(ROOT_PATH, "logs")

# create config subclass to override configuration
class animeTrainingConfig(config.Config):
    # name of the config
    NAME = "animeFace_Faster"
    # since I have 3080ti with 12GB memory, I used 4 images per batch
    # however if the performance is bad then turn down for less images 
    IMAGES_PER_GPU = 4
    # we have 2 classes: BG and animefaces
    NUM_CLASSES = 2
    # training step for each epoch (38/4) (image/batch size)
    STEPS_PER_EPOCH = 10 

    # validation step for each epoch (12/4)
    VALIDATION_STEPS = 3

    #VALIDATION_STEPS = 10
    # detection confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    # reset image size
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

# configuration for testing the model
class animeTestingConfig(animeTrainingConfig):
    # if we use CPU to detect, we set it to 1
    GPU_COUNT = 1
    # we send image 1 at a time
    IMAGES_PER_GPU = 1



# create our own subclass for dataset
class animeDataset(utils.Dataset):

    def load_animedata(self, data_path, subset, labelpath):
        # add anime class
        self.add_class('animeface', 1, "animeface")
        # all picture path
        allpic_path = os.path.join(data_path, subset)
        # label path
        label_path = os.path.join(data_path, labelpath)
        annotation = []
        for file in os.listdir(label_path):
            partial_label = json.load(open(os.path.join(label_path, file)))
            annotation.extend(partial_label)

        # now add images
        for label in annotation:
            # every polygon mask for the anime faces
            polygons = [l['shape_attributes'] for l in label['regions'].values()]
            
            image_path = os.path.join(allpic_path, label['filename'])
            image = skimage.io.imread(image_path)
            # for this project the training image should always be 512*512
            r, c = image.shape[:2]
            
            self.add_image('animeface', image_id=label['filename'], path=image_path,
                           height=r, width=c, polygons=polygons)  

    # This function could generate mask from label for image
    def load_mask(self, image_id):
        # load anime image info
        image_info = self.image_info[image_id]

        # now we convert polygon to bitmap mask
        # 1. create matrix with same size of image, and create n masks corresponding
        #  with number of labels in 1 image
        mask = np.zeros((image_info['height'], image_info['width'], len(image_info['polygons'])),dtype=np.uint8)
        # 2. draw mask with scikit-image
        for num, points in enumerate(image_info['polygons']):
            row, col = skimage.draw.polygon(points['all_points_x'], points['all_points_y'])
            mask[row, col, num] = 1
        # return mask and classID, since we only got 2 classes (BG & face) we return 1s
        return mask, np.ones([mask.shape[-1]], np.int32)
    
    # return image path
    def image_reference(self, image_id):
        return self.image_info[image_id]['path']

# training function
def train(model):

    dataset_train = animeDataset()
    dataset_train.load_animedata('./animedata_faster', 'train', 'train_label')
    # prepare dataset to run
    dataset_train.prepare()


    dataset_val = animeDataset()
    dataset_val.load_animedata('./animedata_faster', 'val', 'val_label')
    dataset_val.prepare()

    print('training with network heads')
    model.train(dataset_train, dataset_val, learning_rate=config.Config.LEARNING_RATE, epochs=30, layers='heads')

# create filter on image
def color_filter(model, image_path="", image_id=uuid.uuid4()):
    print("analysis image on {image}".format(image=image_path))

    image = skimage.io.imread(image_path)
    # convert image to 3 channel RGB
    gray = (skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255).astype(np.uint8)

    # intialize mask to red
    mask_color = gray.copy()
    mask_color[:, : , 0] = 255

    # empty list for cropped images for Resnet and VGG
    crop_img_Faster_RCNN_Res_Vgg = []
    crop_img_Faster_RCNN_V4 = [] # and for V4
    # detect faces
    out = model.detect([image], verbose=1)[0]

    # combine all masks (-1 is last index)
    #mask = (np.sum(out['masks'], -1, keepdims=True) >= 1)

    # if the roi list is not empty then draw bounding box
    if out['rois'].shape[0] > 0:
        splash_2 = image.copy()
        for i in out['rois']:
            start = (i[0], i[1])
            end = (i[2], i[3])
            bounding_box = skimage.draw.rectangle_perimeter(start, end=end, shape=image.shape)
            image_box = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            image_box[bounding_box[0], bounding_box[1]] = 1
            # draw bounding box on the origin image
            splash_2[bounding_box[0], bounding_box[1]] = [255, 110, 90]
            # crop images for step2 to detect
            crop_img = crop_img_coord(image, i)
            #width = np.arange(i[1], i[3])
            #height = np.arange(i[0], i[2])
            #crop_img = image[height, :][:, width]
            resize_img = skimage.transform.resize(crop_img, (224, 224), anti_aliasing=True)
            crop_img_Faster_RCNN_Res_Vgg.append((resize_img*255).astype(np.uint8))
            resize_img = skimage.transform.resize(crop_img, (299, 299), anti_aliasing=True)
            crop_img_Faster_RCNN_V4.append((resize_img*255).astype(np.uint8))
    else:
        splash_2 = image.copy()

    # constant filename
    filename = "./temp_img/{uuid}".format(uuid = image_id)
    png_file_roi = filename+"_faster.png"
    # generate 2 images for display
    skimage.io.imsave(png_file_roi, splash_2)
    # generate images for 2nd step
    # Resnet & VGG
    for i, img in enumerate(crop_img_Faster_RCNN_Res_Vgg):
        png_file_VGG_RES = filename+"_VGG_RES_Faster_"+str(i)+".png"
        skimage.io.imsave(png_file_VGG_RES, img)
    # InceptionV4
    for i, img in enumerate(crop_img_Faster_RCNN_V4):
        png_file_V4 = filename+"_V4_Faster_"+str(i)+".png"
        skimage.io.imsave(png_file_V4, img)

    # return Roi to draw text
    return [out['rois'], out["scores"]]


# evalution function
def detection_evalution(model, image_path=""):

    image = skimage.io.imread(image_path)

    # detect faces
    out = model.detect([image], verbose=1)[0]

    # return Roi to draw text
    return out['rois']

# crop image by coordinates
def crop_img_coord(image, coord):
    width = np.arange(coord[1], coord[3])
    height = np.arange(coord[0], coord[2])
    crop_img = image[height, :][:, width]
    return crop_img

# default setting for back_end
def init_model(weight_path):
    configs = animeTestingConfig()
    model = ml.MaskRCNN(mode='inference', config=configs, model_dir=LOGDIR)
    model.load_weights(weight_path, by_name=True)
    return model

if __name__ == '__main__':

    # usage: (for training) animeface.py train -c --weights=/path/To/weight
    # (for detect) animeface.py detect --weights=/path/To/weight --image=/path/to/image
    parser = argparse.ArgumentParser(prog='animeface', description='Train and run mask RCNN to detect anime faces')
    parser.add_argument('action', help="'train' or 'detect'")
    parser.add_argument('--image', help="the image use for detect faces", required=False, metavar="/path/to/anime/image")
    parser.add_argument('--log', help="path to store checkpoints", required=False,
                        default=LOGDIR, metavar="/path/to/anime/image")
    parser.add_argument('--weights', metavar="/path/to/weights.h5",help=".h5 file only")
    parser.add_argument('-c','--coco', default=False, action='store_true',help="exclude classes")
    parser.add_argument('-w','--without_weight', default=False, action='store_true',help="training without weight")


    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Logs: ", args.log)


    if args.action == 'train':
        configs = animeTrainingConfig()
    elif args.action == 'detect':
        assert args.image, "Please provide image URL"
        configs = animeTestingConfig()

    configs.display()

    if args.action == 'train':
        model = ml.MaskRCNN(mode='training', config=configs, model_dir=args.log)
    elif args.action == 'detect':
        model = ml.MaskRCNN(mode='inference', config=configs, model_dir=args.log)
    # load weight path
    weight_path = args.weights

    if args.action == 'detect':
        model.load_weights(weight_path, by_name=True)

    # doing mitigate learning
    if args.coco == True and args.action == 'train':
        model.load_weights(weight_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif args.without_weight == False and args.action == 'train':
        model.load_weights(weight_path, by_name=True)
    elif args.action == 'train':
        pass # training without loading any weight (Not recommended! The model is too big, taking too much GPU RAM, Loss will be high)

    # Train or evaluate
    if args.action == "train":
        train(model)
    elif args.action == "detect":
        color_filter(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.action))
    print("Finished running")