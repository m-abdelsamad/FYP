# FYP Prohibited Item Detection

This repository contains the code for the final year project on detecting prohibited items in airport security X-ray images using advanced deep learning techniques. Below you will find instructions to set up your environment, prepare your dataset, and run the model.

## Environment Setup

Follow these steps to set up your environment:

```bash
# Create a directory to install Miniconda
mkdir -p ~/miniconda3

# Download the latest Miniconda version
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Run the install script
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Delete the install script
rm -rf ~/miniconda3/miniconda.sh

# Add a conda initialize to your bash
~/miniconda3/bin/conda init bash

# Create a new conda environment
conda create -n pid_env python=3.8

# Activate the environment
conda activate pid_env 

# Install the required Python packages
pip install -r requirements.txt
```

## Data Configuration

To use the models, add your dataset to the `data` directory and update the PIDray yaml file in the `dataConfig` directory to include the paths of the train and validation data. Below is an example configuration for the `PIDray.yaml`:

```bash
# Images and labels directory should be relative to train.py
TRAIN_DIR_IMAGES: ../data/train/images
TRAIN_DIR_LABELS: ../data/train/annotations
VALID_DIR_IMAGES: ../data/val/images
VALID_DIR_LABELS: ../data/val/annotations

# Class names
CLASSES: [
    '__background__',
    'Baton',
    'Pliers',
    'Hammer',
    'Powerbank',
    'Scissors',
    'Wrench',
    'Gun',
    'Bullet',
    'Sprayer',
    'HandCuffs',
    'Knife',
    'Lighter'
]

# Number of classes (object classes + 1 for background class in Faster RCNN)
NC: 13

# Whether to save the predictions of the validation set while training
SAVE_VALID_PREDICTION_IMAGES: True
```

Update the names of the classes and the number of classes if you are using a different dataset than the PIDray dataset.

## Dataset Annotations Setup

The dataset annotations need to be in a specific XML format for compatibility with the training pipeline. Below is an example of how the annotations are expected to look:

```xml
<annotation>
    <folder>/data/test/images</folder>
    <filename>image0001.png</filename>
    <path>/data/test/images/image0001.png</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>560</width>
        <height>448</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>Gun</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <occluded>0</occluded>
        <bndbox>
            <xmin>309</xmin>
            <ymin>207</ymin>
            <xmax>494</xmax>
            <ymax>249</ymax>
        </bndbox>
    </object>
</annotation>
```

Important Annotation Elements

- `<bndbox>`: Contains the coordinates for the bounding box.
- `<object><name></name></object>`: Contains the name of the object to detect.
- `<size>`: Contains the size of the image (width, height, depth).
- `<folder>`, `<filename>`, and `<path>`: Contain information about the name and location of the image.

The other elements can have arbitrary values if not included with in your dataset annotations.


### Conversion from JSON to XML

The PIDray dataset annotations do not originally come in this XML format but are provided in a single JSON file. To convert these annotations to the required XML format, use the `XmlAnnotationConverter.py` script in the repository.

Specify the location of the JSON annotations file on line 64 of the script:

```python
annotations_path = '/data_configs/xray_train.json'
```

Update the location of the images downloaded from the PIDray dataset on line 102:

```python
original_images_folder = './data/pidimages/train'
```

After updating these values, execute the script to generate the annotations in the required format. This process will create XML files that match the required structure for training the models.

## Running the Models

To run the model against the data configured in the PIDray.yaml file, use the following command:

```bash
python train.py --data /data_configs/PIDray.yaml --epochs 10 --model fasterrcnn_resnet50 --name /path/to/output/the/model --batch 32
```

- `--data`: This parameter requires the path to the dataset configuration file. The configuration file should specify paths to training and validation data, as well as other settings related to data handling. For example: `--data /data_configs/PIDray.yaml`.

- `--model`: This parameter specifies the name of the model you want to use. You can choose from the following models:
    - `fasterrcnn_resnet50_fpn`: Faster R-CNN with a ResNet-50 backbone and Feature Pyramid Networks.
    - `fasterrcnn_resnet50`: Faster R-CNN with a ResNet-50 backbone.
    - `fasterrcnn_resnet50_cbam_w_aspp`: Faster R-CNN with a ResNet-50 backbone, enhanced with Convolutional Block Attention Module (CBAM) and Atrous Spatial Pyramid Pooling (ASPP).

- `--name`: This parameter defines the location where you want to save the model information. This includes weights for the model and any results from the training process. It should be a path to the directory where these files will be stored, for example: `--name /path/to/output/the/model`.

Ensure these parameters are correctly configured to effectively run your training script and manage your models and data.

For a list of additional parameters, check the top of the `train.py` file for more information.

The terimal output should be similar to the following:

```bash
Number of training samples: 665
Number of validation samples: 72

3,191,405 total parameters.
3,191,405 training parameters.
Epoch     0: adjusting learning rate of group 0 to 1.0000e-03.
Epoch: [0]  [ 0/84]  eta: 0:02:17  lr: 0.000013  loss: 1.6518 (1.6518)  time: 1.6422  data: 0.2176  max mem: 1525
Epoch: [0]  [83/84]  eta: 0:00:00  lr: 0.001000  loss: 1.6540 (1.8020)  time: 0.0769  data: 0.0077  max mem: 1548
Epoch: [0] Total time: 0:00:08 (0.0984 s / it)
creating index...
index created!
Test:  [0/9]  eta: 0:00:02  model_time: 0.0928 (0.0928)  evaluator_time: 0.0245 (0.0245)  time: 0.2972  data: 0.1534  max mem: 1548
Test:  [8/9]  eta: 0:00:00  model_time: 0.0318 (0.0933)  evaluator_time: 0.0237 (0.0238)  time: 0.1652  data: 0.0239  max mem: 1548
Test: Total time: 0:00:01 (0.1691 s / it)
Averaged stats: model_time: 0.0318 (0.0933)  evaluator_time: 0.0237 (0.0238)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.009
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.007
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.074
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.088
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.167
SAVING PLOTS COMPLETE...
...
Epoch: [4]  [ 0/84]  eta: 0:00:20  lr: 0.001000  loss: 0.9575 (0.9575)  time: 0.2461  data: 0.1662  max mem: 1548
Epoch: [4]  [83/84]  eta: 0:00:00  lr: 0.001000  loss: 1.1325 (1.1624)  time: 0.0762  data: 0.0078  max mem: 1548
Epoch: [4] Total time: 0:00:06 (0.0801 s / it)
creating index...
index created!
Test:  [0/9]  eta: 0:00:02  model_time: 0.0369 (0.0369)  evaluator_time: 0.0237 (0.0237)  time: 0.2494  data: 0.1581  max mem: 1548
Test:  [8/9]  eta: 0:00:00  model_time: 0.0323 (0.0330)  evaluator_time: 0.0226 (0.0227)  time: 0.1076  data: 0.0271  max mem: 1548
Test: Total time: 0:00:01 (0.1116 s / it)
Averaged stats: model_time: 0.0323 (0.0330)  evaluator_time: 0.0226 (0.0227)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.137
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.313
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.175
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.204
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.683
SAVING PLOTS COMPLETE...
```

