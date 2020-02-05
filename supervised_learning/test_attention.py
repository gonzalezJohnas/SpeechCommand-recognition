import argparse

# Data Loading
from tensorflow.python.keras import Model

from tensorflow.keras.backend import squeeze

from global_utils import read_sample
import numpy as np
import matplotlib.pyplot as plt
from supervised_learning.config import *
import random
from scipy import signal
import scipy.io.wavfile as wavfile

random.seed(42)


def main(args):
    print("Loading the model")
    model = tf.keras.models.load_model(args.model, custom_objects={
        'squeeze': squeeze}
                                       )
    model.summary()

    sample_mfcc = read_sample(args.filename, mfcc=True)

    x_test = np.expand_dims(sample_mfcc, axis=0)
    x_test = np.expand_dims(x_test, -1)

    attSpeechModel = Model(inputs=model.input,
                           outputs=[model.get_layer('output').output,
                                    model.get_layer('attSoftmax').output])

    predictions, attention_weights = attSpeechModel.predict(x_test)

    prediction_max = tf.argmax(predictions, axis=1).numpy()
    print("Label predicted {}".format(id2name[prediction_max[0]]))
    print("Max predictions {}".format(np.max(predictions)))

    imgHeight = 2 * 2

    _, wav = wavfile.read(args.filename)

    # plot the first 1024 samples
    plt.figure(figsize=(17, imgHeight))
    plt.plot(wav)
    # label the axes
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    # set the title
    plt.title("Audio sample")
    # display the plot

    plt.figure(figsize=(17, imgHeight))
    plt.title('Attention weights (log)')
    plt.ylabel('Log of attention weight')
    plt.xlabel('Mel-spectrogram index')
    plt.plot(np.log(attention_weights[0]))

    f, t, Sxx = signal.spectrogram(wav, SAMPLE_RATE)

    plt.figure(figsize=(17, imgHeight))
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 1000.0)

    plt.show()

    return 1


if __name__ == "__main__":
    print("TensorFlow version: {}".format(tf.__version__))

    print("Eager execution: {}".format(tf.executing_eagerly()))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        help='Path of a previous trained model',
        required=True,
        default=None
    )

    parser.add_argument(
        '--filename',
        help='Wav file name to test',
        required=True,
        default=None
    )

    args = parser.parse_args()

    main(args)
