
import argparse

# Data Loading
import pickle

from tensorflow.keras.backend import squeeze

from utils import detect_speech, get_dataset_files, format_dataset
from config import *

import os
import numpy as np
import tqdm
import random

random.seed(42)


def main(args):
    print("Loading the model")
    model = tf.keras.models.load_model(args.model, custom_objects={'squeeze': squeeze})

    if args.filename:
        prediction_max_index, probabilities = detect_speech(model, args.filename)

        print("Label predicted {}".format(id2name[prediction_max_index[0]]))
        print("Max predictions {}".format(np.max(probabilities)))

        return 1

    if args.serialize:
        testset = pickle.load(open(os.path.join(args.indir, 'testset.p'), 'rb'))

    else:
        print("Loading wave file")
        testset = get_dataset_files(args.indir, list_files="testing_list.txt")
        pickle.dump(testset, open("data/testset.p", "wb"))

    print("The test dataset have {} samples".format(len(testset)))

    dataset_test = tf.data.Dataset.from_tensor_slices(format_dataset(testset)).batch(1)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    for x_test, y_test in tqdm.tqdm(dataset_test, total=len(testset)):
        predictions = model(x_test, training=False)
        test_accuracy(y_test, predictions)

    print("Test Accuracy: {}".format(test_accuracy.result() * 100))


if __name__ == "__main__":
    print("TensorFlow version: {}".format(tf.__version__))

    print("Eager execution: {}".format(tf.executing_eagerly()))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--indir',
        help='Aboslute path to data directory containing .wav files',
        required=False
    )

    parser.add_argument(
        '--serialize',
        help='Loading from serialize object',
        required=False,
        default=False
    )

    parser.add_argument(
        '--model',
        help='Checkpoint path of a previous trained model',
        required=False,
        default=None
    )


    parser.add_argument(
        '--filename',
        help='Wav file to test',
        required=False,
        default=None
    )

    args = parser.parse_args()

    main(args)




