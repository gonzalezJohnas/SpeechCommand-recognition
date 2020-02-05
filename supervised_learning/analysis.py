from global_utils  import *

# target word
TARGET_WORD = 'right'

def display_lowpass_normal(wav, lowpass_signal, fs, label=''):


    fig, (axs_raw, axs_low) = plt.subplots(2)
    fig.tight_layout(pad=3.0)
    fig.set_figheight(FIG_HEIGHT)
    fig.set_figwidth(FIG_WIDTH)

    # display the plot
    axs_raw.plot(wav)
    # label the axes
    axs_raw.set_ylabel("Amplitude", fontsize=FONT_SIZE)
    axs_raw.set_xlabel("Time", fontsize=FONT_SIZE)
    # set the title
    axs_raw.set_title("Audio sample : {}".format(label), fontsize=FONT_SIZE)

    axs_low.plot(lowpass_signal)
    # label the axes
    axs_low.set_ylabel("Amplitude", fontsize=FONT_SIZE)
    axs_low.set_xlabel("Time", fontsize=FONT_SIZE)
    # set the title
    axs_low.set_title("Audio sample with low pass filter", fontsize=FONT_SIZE)

    f_raw, periodogram_raw = signal.periodogram(wav, fs)
    f_raw, periodogram_low = signal.periodogram(lowpass_signal, fs)

    fig, (axs_periodogram_raw, axs_periodogram_low) = plt.subplots(2)
    fig.tight_layout(pad=3.0)
    fig.set_figheight(FIG_HEIGHT)
    fig.set_figwidth(FIG_WIDTH)


    axs_periodogram_raw.semilogy(f_raw, periodogram_raw)
    axs_periodogram_raw.set_xlabel('frequency [Hz]', fontsize=FONT_SIZE)
    axs_periodogram_raw.set_ylabel('PSD [V**2/Hz]', fontsize=FONT_SIZE)
    axs_periodogram_raw.set_title("Periodogram raw signal", fontsize=FONT_SIZE)

    axs_periodogram_low.semilogy(f_raw, periodogram_low)
    axs_periodogram_low.set_xlabel('frequency [Hz]', fontsize=FONT_SIZE)
    axs_periodogram_low.set_ylabel('PSD [V**2/Hz]', fontsize=FONT_SIZE)
    axs_periodogram_low.set_title("Periodogram low pass filtered signal", fontsize=FONT_SIZE)


def main(args):
    if args.wavfile:
        fs, wav = wavfile.read(args.wavfile, "wb")
        lowpass_signal = low_pass_filter(wav, sample_rate=fs, cutoff_frequency=1000)

        display_lowpass_normal(wav, lowpass_signal, fs)
        plt.show()

    elif args.indir:
        data_dict = get_data(args.indir)
        word_samples = data_dict[TARGET_WORD]

        mean_lowpass_array, normal_array = mean_low_pass_filter(word_samples, SAMPLE_RATE, CUTOFF_FREQ)
        display_lowpass_normal(normal_array, mean_lowpass_array, SAMPLE_RATE, TARGET_WORD)
        plt.show()





    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--wavfile',
        help='Path to the .wav files',
        required=False
    )

    parser.add_argument(
        '--indir',
        help='Absolute path to data directory containing .wav files',
        required=False
    )

    args = parser.parse_args()
    main(args)
