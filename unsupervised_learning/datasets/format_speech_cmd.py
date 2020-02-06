import sys
sys.path.append("..")

from global_utils import get_data_speaker, low_pass_filter
import argparse
from shutil import copyfile
import os
import tqdm
import scipy.io.wavfile as wavfile
from config import *
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir',
        help='Aboslute path to data directory of the speech command dataset',
        required=True
    )

    parser.add_argument(
        '--output_dir',
        help='output dir name',
        required=True,
    )

    parser.add_argument(
        '--low_pass',
        help='if performing low_pass filter on the input data',
        action='store_true'
    )

    args = parser.parse_args()

    os.mkdir(args.output_dir)

    speaker_dic = get_data_speaker(args.input_dir)

    for speaker_id in tqdm.tqdm(speaker_dic.keys()):
        speaker_dir = os.path.join(args.output_dir, speaker_id)
        os.mkdir(speaker_dir)

        for wav_name in speaker_dic[speaker_id]:
            wav_new_name = wav_name[0].split('/')[-1].split('.')[0]
            _, wav = wavfile.read(wav_name[0], "wb")
            dest_path = os.path.join(speaker_dir, f"{wav_new_name}_{wav_name[1]}.wav")

            if args.low_pass:
                low_pass_signal = low_pass_filter(wav, SAMPLE_RATE , CUTOFF_FREQ )
                low_pass_signal = low_pass_signal + 100
                wavfile.write(dest_path, SAMPLE_RATE, np.asarray(low_pass_signal, dtype=np.int16))
            else:
                copyfile(wav_name[0], dest_path)

