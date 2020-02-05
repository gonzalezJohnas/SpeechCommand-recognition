import sys
sys.path.append("..")

from global_utils import get_data_speaker
import argparse
from shutil import copyfile
import os
import tqdm


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

    args = parser.parse_args()

    os.mkdir(args.output_dir)

    speaker_dic = get_data_speaker(args.input_dir)

    for speaker_id in tqdm.tqdm(speaker_dic.keys()):
        speaker_dir = os.path.join(args.output_dir, speaker_id)
        os.mkdir(speaker_dir)

        for wav_name in speaker_dic[speaker_id]:
            wav_new_name = wav_name[0].split('/')[-1].split('.')[0]
            dest_path = os.path.join(speaker_dir, f"{wav_new_name}_{wav_name[1]}.wav")
            copyfile(wav_name[0], dest_path)

        print(f"Finished processing speaker {speaker_id}")
