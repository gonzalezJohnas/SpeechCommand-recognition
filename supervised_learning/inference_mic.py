import argparse
import tensorflow as tf
# Data Loading
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
from tensorflow.keras.backend import squeeze

from supervised_learning.config import *
from global_utils import id2name
import random

random.seed(42)

threshold = 0.7


def detect_speech(model, wav_file):

    """
    Wrapper function to run inference on a fivel model and a given audio filename
    :param model: Tensorflow model to run inferennce on
    :param wav_file: Filename of the audio file
    :return:
    """
    _, wav = wavfile.read(wav_file, "wb")

    wav = wav.astype(np.float32)
    x_test = get_MFCC(wav)
    x_test = np.expand_dims(x_test, axis=0)
    x_test = np.expand_dims(x_test, -1)

    predictions = model.predict(x_test)
    prediction_max_index = tf.argmax(predictions, axis=1).numpy()
    proba = np.max(predictions)


if __name__ == "__main__":
    print("TensorFlow version: {}".format(tf.__version__))

    print("Eager execution: {}".format(tf.executing_eagerly()))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        help='Checkpoint path of a previous trained model',
        required=True,
        default=None
    )

    args = parser.parse_args()

    print("Loading the model")
    model = tf.keras.models.load_model(args.model, custom_objects={
        'squeeze': squeeze}
                                       )
    model.summary()



    while True:

        myrecording = sd.rec(int(LENGTH_INPUT * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()  # Wait until recording is finished
        wavfile.write('/tmp/output.wav', SAMPLE_RATE, myrecording)  # Save as WAV file

        prediction_max_index, proba = detect_speech(model, '/tmp/output.wav')

        if np.max(proba) > threshold:
            print("Label predicted {}".format(id2name[prediction_max_index[0]]))
            print("Max predictions {}".format(np.max(proba)))
