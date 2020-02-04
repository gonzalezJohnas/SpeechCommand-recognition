import tensorflow as tf
import datetime
import os 
"""
CONFIG FILE FOR GLOBAL VARIABLES
"""
# GLOBAL VARIABLES FOR PREPROCESSING
LABELS = 'yes no up down left right on off stop go unknown'.split()
SAMPLE_RATE = 16000
LENGTH_INPUT = 1  # Duration of recording

id2name = {i: name for i, name in enumerate(LABELS)}
name2id = {name: i for i, name in id2name.items()}
NUM_MFCC=26
CUTOFF_FREQ=700
NB_UNKNOWN_CLASS = 2000
# GLOBAL VARIABLES FOR TRAINING
BATCH_SIZE = 16
EPOCHS = 40
LR_INIT = 0.01

# TENSORBOARD GLOBAL VARIABLES
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/gradient_tape/' + current_time + '/train'
checkpoint_path = "logs/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# PLOT CONFIG
FIG_HEIGHT=20
FIG_WIDTH=30
FONT_SIZE=40