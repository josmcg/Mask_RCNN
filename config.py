from mrcnn.config import Config
from utils.voc_utils import ICDAR_convert

class PageConfig(Config):
    NAME = "PAGES"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3
    IMAGE_MIN_DIM = 1920
    IMAGE_MAX_DIM = 1920
    RPN_ANCHOR_SCALES = (128,265,512,768)
    TRAIN_ROIS_PER_IMAGE = 5
    STEPS_PER_EPOCH = 2
    VALIDATION_STEPS = 3
    NUM_CLASSES = len(ICDAR_convert.keys()) + 1 # background class


