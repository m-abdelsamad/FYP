import pandas as pd
import json
import xml.etree.ElementTree as ET
from PIL import Image
import os
import shutil
from tqdm import tqdm
import random

def create_xml_files(annotations_table, images_folder, category_mapping, output_image_folder, output_xml_folder):
    # Create directories if they don't exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_xml_folder):
        os.makedirs(output_xml_folder)

    for index, row in tqdm(annotations_table.iterrows(), total=annotations_table.shape[0]):
        file_name = row['file_name']
        image_path = os.path.join(images_folder, file_name)

        # Using Pillow to get image size
        with Image.open(image_path) as img:
            width, height = img.size

        # Copy image to the corresponding folder
        shutil.copy(image_path, os.path.join(output_image_folder, file_name))

        # XML file creation
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = output_image_folder
        ET.SubElement(root, "filename").text = file_name
        ET.SubElement(root, "path").text = os.path.join(output_image_folder, file_name)

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "roboflow.ai"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(root, "segmented").text = "0"

        obj = ET.SubElement(root, "object")
        category_name = category_mapping.get(row['category_id'], 'Unknown')
        ET.SubElement(obj, "name").text = category_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "occluded").text = "0"

        bbox = row['bbox']
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
        ET.SubElement(bndbox, "xmax").text = str(int(bbox[0] + bbox[2]))
        ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
        ET.SubElement(bndbox, "ymax").text = str(int(bbox[1] + bbox[3]))

        tree = ET.ElementTree(root)
        xml_filename = os.path.join(output_xml_folder, f"{file_name.split('.')[0]}.xml")
        tree.write(xml_filename)

# Define the path to the annotations file
annotations_path = '/data_configs/xray_train.json'

# Load the annotations
with open(annotations_path) as json_file:
    annotations = json.load(json_file)

image_dict = {image['id']: {'file_name': image['file_name']} for image in annotations['images']}
table_data = []

for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    if image_id in image_dict:
        file_name = image_dict[image_id]['file_name']
        category_id = annotation['category_id']
        bbox = annotation['bbox']

        table_data.append({
            'file_name': file_name,
            'category_id': category_id,
            'bbox': bbox
        })

annotations_table = pd.DataFrame(table_data)
category_mapping = {category['id']: category['name'] for category in annotations['categories']}
annotations_table = annotations_table.sample(frac=1).reset_index(drop=True)

# Split the shuffled DataFrame
split_index = int(len(annotations_table) * 0.9)
train_annotations_table = annotations_table[:split_index]
val_annotations_table = annotations_table[split_index:]


# Paths to training and validation folders
train_images_folder = './data/train/images'
val_images_folder = './data/val/images'
train_annotations_folder = './data/train/annotations'
val_annotations_folder = './data/val/annotations'

original_images_folder = './data/pidimages/train'

# Create XML files and organize images for training and validation sets
create_xml_files(train_annotations_table, original_images_folder, category_mapping, train_images_folder, train_annotations_folder)
create_xml_files(val_annotations_table, original_images_folder, category_mapping, val_images_folder, val_annotations_folder)

print("Training and validation sets created.")
