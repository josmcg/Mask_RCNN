from mrcnn.config import Config
from utils.voc_utils import ICDAR_convert

classes = ["Figure", "Table", "Equation"]
          
class PageConfig(Config):
    NAME = "PAGES"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 1484
    IMAGE_MAX_DIM = 1920
    RPN_ANCHOR_SCALES = (64,128,256,512,768)
    TRAIN_ROIS_PER_IMAGE = 7
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 100
    NUM_CLASSES = len(classes) + 1 # background class
    
