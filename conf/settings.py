"""
author:guopei
"""
from datetime import datetime


TRAIN_MEAN = [0.485, 0.499, 0.432]
TRAIN_STD = [0.232, 0.227, 0.266]


DATA_PATH = '/home/py/code/JDAI/fast-reid/datasets/PA-100K/annotation'
IMAGE_PATH = '/home/py/code/JDAI/fast-reid/datasets/PA-100K/data'
MILESTONES = [20, 20, 45]

#weights file directory
CHECKPOINT_PATH = 'checkpoints'

TIME_NOW = datetime.now().isoformat()

#tensorboard log file directory
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 5

#input image size for network:[width, height]
IMAGE_SIZE = (128,256)
