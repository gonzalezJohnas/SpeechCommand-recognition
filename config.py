import tensorflow as tf
import datetime

"""
CONFIG FILE FOR GLOBAL VARIABLES
"""
# GLOBAL VARIABLES FOR PREPROCESSING
LABELS = 'yes no up down left right on off stop go unknown'.split()

SAMPLE_RATE = 16000
LENGTH_INPUT = 1  # Duration of recording

id2name = {i: name for i, name in enumerate(LABELS)}
name2id = {name: i for i, name in id2name.items()}


# GLOBAL VARIABLES FOR TRAINING
BATCH_SIZE = 16
EPOCHS = 40
LR_INIT = 0.01


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    LR_INIT,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

optimizer_adam = tf.keras.optimizers.Adagrad(learning_rate=LR_INIT)

# Define our metrics

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
validation_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)
validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('validation_accuracy')

# TENSORBOARD GLOBAL VARIABLES
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/gradient_tape/' + current_time + '/train'
checkpoint_path = "logs/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)