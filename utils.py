from __future__ import absolute_import, division, print_function, unicode_literals

import librosa.display
import librosa
import numpy as np
import scipy.io.wavfile as wavfile

# Data Loading
import os
import re
from glob import glob
import tqdm
import matplotlib.pyplot as plt

from config import *
import tensorflow as tf
import random



def format_dataset(dataset):
    """
    Format a dataset for the tensorflow API
    :param dataset: list of samples
    :return:
    """
    inputs = []
    labels = []
    random.shuffle(dataset)

    for x in dataset:
        inputs.append(np.expand_dims(x[2], -1))
        labels.append(x[0])

    return inputs, labels

def read_sample(wav_file, sample_rate=16000, num_mfcc=None):
    """
    From a file name return the raw audio data or the mfcc features
    :param wav_file: file path of the file
    :param sample_rate: sampling rate of the audio signal
    :param num_mfcc: if set compute MFCC features with num_fcc
    :return: np array of raw signal or MFCC features
    """
    _, wav = wavfile.read(wav_file, "wb")

    if len(wav) < sample_rate:
        # pad to one second
        # (note that in contrast to the original kernel)
        wav = np.pad(wav, (0, sample_rate - len(wav)), 'median')
    elif len(wav) > sample_rate:
        # ignore noise samples
        return np.ndarray((1, 1), dtype=np.float32)

    wav = wav.astype(np.float32)
    if num_mfcc:
        wav = get_MFCC(wav, sample_rate, nb_mfcc_features=num_mfcc)

    return wav


def get_MFCC(sample, sample_rate=16000, nb_mfcc_features=26):
    """
    Use librosa to compute MFCC features from an audio array with sample rate and number_mfcc
    :param sample:
    :param sample_rate:
    :param nb_mfcc_features:
    :return: np array of MFCC features
    """
    mfcc_feat = librosa.feature.mfcc(sample, sr=sample_rate, n_mfcc=nb_mfcc_features)
    mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat


def get_dataset_files(data_dir, list_files=None):
    """
    Retrieve all *.wav file under the data_dir directory. If list_files is set will only return the filenames
    inside list_files.
    :param data_dir: Root path of the data directory
    :param list_files: Filename of the list of files to return
    :return: list of filenames
    """

    dataset = []
    uid_fileset = set()
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    selected_files = []

    if (list_files):
        with open(os.path.join(data_dir, list_files), 'r') as fin:
            selected_files = fin.read().splitlines()

    possible = set(LABELS)

    for entry in tqdm.tqdm(selected_files):
        entry = os.path.join(data_dir, entry)
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label not in possible:
                continue

            uid_fileset.add(uid)
            label_id = name2id[label]

            audio_input = read_sample(entry, num_mfcc=True)

            if len(audio_input) > 0:
                sample = (label_id, uid, audio_input,
                          # for debugging
                          entry
                          )
                dataset.append(sample)
    return dataset


def load_data(data_dir):
    """
    Load all the data from the data_dir, return three dataset for training (train,val,test)
    :param data_dir:  Root path of the data directory
    :return: list
    """
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, '*/*wav'))

    validation_dataset = get_dataset_files(data_dir, 'validation_list.txt')
    valset = set([item[1] for item in validation_dataset])

    test_dataset = get_dataset_files(data_dir, 'testing_list.txt')
    testset = set([item[1] for item in test_dataset])

    possible = set(LABELS)
    unknown, train, val, test = [], [], [], []

    all_files_set = set(all_files) - (valset + testset)

    for entry in tqdm.tqdm(all_files_set):
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)

            wav = read_sample(entry, sample_rate=SAMPLE_RATE, num_mfcc=True)

        if label not in possible and len(unknown) < 2000:
            # ignore this sample
            label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, wav,
                      # for debugging
                      entry
                      )
            unknown.append(sample)

        else:
            label_id = name2id[label]
            sample = (label_id, uid, wav,
                      # for debugging
                      entry
                      )

            train.append(sample)

    return train + unknown, val, test


def get_data(data_dir):
    """
    Given a data dir read all the files in a dictionnary with class as key
    :param data_dir: path to the speech command dataset
    :return:
    """

    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, '*/*wav'))

    data_files = {}

    for entry in tqdm.tqdm(all_files):
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)

            if label not in data_files.keys():
                data_files[label] = [entry]

            else:
                data_files[label].append(entry)

    return data_files


def visualize_audio(data_path):
    """
    Wrapper function to read and plot a wave file
    :param data_path: path to the wave file
    :return:
    """

    _, wav = wavfile.read(data_path)
    # signal = np.array(np.frombuffer(wav, dtype='UInt8'), dtype='Int8')

    # plot the first 1024 samples
    plt.plot(wav)
    # label the axes
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    # set the title
    plt.title("Audio sample")
    # display the plot
    plt.show()


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

    return prediction_max_index, proba
