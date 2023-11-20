# Deep Learning Classification for Anime Cartoon Faces

### Author: Jiasong Liu, Fangdi Liu, .Ratnaker

In this project, we are going to classify anime faces by doing 2 steps. The first step is to use a Mask/Faster RCNN learning model to recognize if the input image contains anime faces. And the second step is to use 3 differnt deep learning models to classify emotion.

## Running instruction
This program is test and run in windows 10, 11. It is not tested in MacOS or other linux OS.\
Please create a virtual environment with python 3.8.x. We test our program with python 3.8.18\
Then run `pip install -r requirements.txt` in the `Group8_projectCode/` path.

Warning: all weights files are not submitted in the code. Please use python file to train the weights, the default weight path after training complete is `./logs/`. if you want to try our weights. Please download weights at: https://drive.google.com/drive/folders/1zpR9ITruHPd6mpRI0IZCj-0zeU2PQcvb?usp=drive_link \
Please put all weight files in the `Group8_projectCode/logs/` since default weight path is in `logs` folder. Use `--log="path/to/weight"` to override the path. Running without weight can cause unexpected result.

## Warning: Training with or without weight for Faster and Mask RCNN can cause exhaust resource error. Make sure your computer is capable of training before run.

## Project directory
```css
.
├── animeData
│   ├── train
│   ├── train_label
│   ├── val
│   └── val_label
├── animedata_faster
│   ├── train
│   ├── train_label
│   ├── val
│   └── val_label
├── animeFaces
├── logs
├── temp_img
├── Test_set_step1
├── Test_set_step2
└── ...
```

## For GUI Usage
To use the gui, run `python AnimeFaceDetection.py`.

1. Click "Upload Image" to select the anime image that needs to be recognized.
2. Select the recognized face to be processed on the right side of the GUI.
3. In the lower part of the GUI, select the model you want to use to recognize the emotions of the anime character's face.
4. Click "Reset Image" to reselect the image.

## For Terminal Usage
### Step 1:
Two files could be run in step 1: `animeface.py` for train and test Mask RCNN.
`animeface_faster.py` for train and test Faster RCNN. The training and validation directories are default to `./animeData, ./animedata_faster` folder and `./animeFaces` folder
Here are some examples for running `animeface.py`
```bash
# train the model with weight
python animeface.py train --weights=./path/to/weight.h5
# train without model
python animeface.py train -w
# run detection
python animeface.py detect --image=./image/path.jpg --weights=./path/to/weights.h5

```
Here are some examples for running `animeface_faster.py`
```bash
# train the model with weight
python animeface_faster.py train --weights=./path/to/weight.h5
# train without model
python animeface_faster.py train -w
# run detection
python animeface_faster.py detect --image=./image/path.jpg --weights=./path/to/weights.h5

```


### Step 2:

For InceptionV4:
```bash
# train the model
python InceptionV4.py train emotion --weights --source
# detect the model
python InceptionV4.py detect emotion --image --weights
# example
# for detection: python InceptionV4.py detect emotion --image=./abc.jpg --weights=./weights/file/name.keras
# for training: python InceptionV4.py train emotion
```

For ResNet:
```bash
# train the model
python ResNet.py train emotion --log --weights --source --model
# detect the model
python ResNet.py detect emotion --image --weights --model
# example
# for detection: python Resnet.py detect emotion --image=./abc.jpg --weights=./weights/file/name.keras --model=resnet34
# for training: python Resnet.py train emotion --model resnet34 --weights=./weights/file/name.keras
```

For VGG:
```bash
# train the model
python VGG.py train emotion --log --source --model --weights --dropout
# detect the model
python VGG.py detect emotion --image --weights --model --dropout
# example
# for detection: python VGG.py detect emotion --weights=./weights/file/name.keras --image=./abc.jpg --model VGG19
# for training: python VGG.py train emotion --model VGG19 --dropout
```

To get more information, please use 'help' in terminal.


## Evaluation:

To run step 1 model evaluation, input: `python Evaluation_step1.py`. Please ensure the weight files is in `logs` directory with name `Mask_RCNN_animeFace.h5`, `Faster_RCNN_animeFace.h5`.

To run step 2 model evaluation, imput: `python Evaluation_step2.py model_name --model_type --weights=./weights/file/name.keras`. Please ensure the weights are in `logs` directory.

* For example: `python Evaluation_step2.py ResNet --model_type resnet50 --weights=./weights/file/name.keras`




