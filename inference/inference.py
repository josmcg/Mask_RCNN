#!/usr/bin/env python3
"""
Script to run an end to end pipeline
"""

import multiprocessing as mp
from mrcnn.config import Config
from argparse import ArgumentParser
import mrcnn.model as modellib
from config import PageConfig, classes
from dataset.dataset import PageDataset
import os
import subprocess
from model2xml import model2xml
from tqdm import tqdm
import shutil
from utils.voc_utils import ICDAR_convert

# PDF directory path

parser = ArgumentParser(description="Run the classifier")
parser.add_argument('-d', "--weightsdir", default='weights', type=str, help="Path to weights dir")
parser.add_argument('-w', "--weights", type=str, help='Path to weights file', required=True)
parser.add_argument('-t', "--threads", default=160, type=int, help="Number of threads to use")

args = parser.parse_args()

class InferenceConfig(Config):
    NAME = "PAGES"
    backbone = "resnet50"
    GPU_COUNT = 1
    IMAGE_MIN_DIM = 1484
    IMAGE_MAX_DIM = 1920
    RPN_ANCHOR_SCALES = (64,128,256,512,768)
    NUM_CLASSES = len(classes) +1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
config = PageConfig()
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=args.weightsdir)
#model_path = model.find_last()
#print("Loading weights from ", model_path)
model.load_weights(args.weights, by_name=True)
data_test = PageDataset('test', 'ICDAR_data_split', 0, nomask=True)
#data_test.load_page(classes=['Figure', 'Table', 'Equation', 'Body Text'])
data_test.load_page(classes=classes)
data_test.prepare()
image_ids = data_test.image_ids

if not os.path.exists('xml'):
    os.makedirs('xml')

for idx, image_id in enumerate(tqdm(image_ids)):
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(data_test, inference_config,image_id, use_mini_mask=False)
    results = model.detect([image], verbose=0)
    print(results)
    r = results[0]
    info = data_test.image_info[image_id]
    zipped = zip(r["class_ids"], r["rois"])
    model2xml(info["str_id"], 'xml', [1920, 1920], zipped, data_test.class_names, r['scores'])



