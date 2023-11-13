import os
import json
import uuid
from datetime import datetime
import utils
import config
import argparse
import skimage
import numpy as np
import model as ml



# first, let us create label that fit the model
ROOT_PATH = os.getcwd()
LOGDIR = os.path.join(ROOT_PATH, "logs")

# create config subclass to override configuration
class animeTrainingConfig(config.Config):
    # name of the config
    NAME = "animeFace"
    # since I have 3080ti with 12GB memory, I used 4 images per batch
    # however if the performance is bad then turn down for less images 
    IMAGES_PER_GPU = 4
    # we have 2 classes: BG and animefaces
    NUM_CLASSES = 2
    # training step for each epoch (1055/4) (image/batch size)
    STEPS_PER_EPOCH = 264 

    # validation step for each epoch (140/4)
    VALIDATION_STEPS = 35

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
    dataset_train.load_animedata('./animedata', 'train', 'train_label')
    # prepare dataset to run
    dataset_train.prepare()


    dataset_val = animeDataset()
    dataset_val.load_animedata('./animedata', 'val', 'val_label')
    dataset_val.prepare()

    print('training with network heads')
    model.train(dataset_train, dataset_val, learning_rate=config.Config.LEARNING_RATE, epochs=30, layers='heads')

# create filter on image
def color_filter(model, image_path=""):
    print("analysis image on {image}".format(image=image_path))

    image = skimage.io.imread(image_path)
    # convert image to 3 channel RGB
    gray = (skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255).astype(np.uint8)

    # detect faces
    out = model.detect([image], verbose=1)[0]
    # combine all masks (-1 is last index)
    mask = (np.sum(out['masks'], -1, keepdims=True) >= 1)

    # color image if mask is not 0
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray

    # if the roi list is not empty then draw bounding box
    if out['rois'].shape[0] > 0:
        for i in out['rois']:
            start = (i[0], i[1])
            end = (i[2], i[3])
            bounding_box = skimage.draw.rectangle_perimeter(start, end=end, shape=image.shape)
            image_box = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            image_box[bounding_box[0], bounding_box[1]] = 1
            splash[bounding_box[0], bounding_box[1]] = [255, 110, 90]

    # unique filename
    filename = "{uuid}_{date:%Y%m%d}".format(uuid = uuid.uuid4(), date = datetime.now())
    png_file = filename+".png"
    skimage.io.imsave(png_file, splash)

if __name__ == '__main__':

    # usage: (for training) animeface.py train -c --weights=/path/To/weight
    # (for detect) animeface.py detect --weights=/path/To/weight --image=/path/to/image
    parser = argparse.ArgumentParser(prog='animeface', description='Train and run mask RCNN to detect anime faces')
    parser.add_argument('action', help="'train' or 'detect'")
    parser.add_argument('--image', help="the image use for detect faces", required=False, metavar="/path/to/anime/image")
    parser.add_argument('--log', help="path to store checkpoints", required=False,
                        default=LOGDIR, metavar="/path/to/anime/image")
    parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5",help=".h5 file only")
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

    # doing mitigate learning
    if args.coco == True:
        model.load_weights(weight_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif args.without_weight == False:
        model.load_weights(weight_path, by_name=True)
    else:
        pass # training without loading any weight (Not recommended! The model is too big, taking too much GPU RAM, Loss will be high)

    # Train or evaluate
    if args.action == "train":
        train(model)
    elif args.action == "detect":
        color_filter(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.action))

print()
print(ROOT_PATH)