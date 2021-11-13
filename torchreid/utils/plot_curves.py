import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse
import sys


def get_data(tfevents_path):
    ea = event_accumulator.EventAccumulator(tfevents_path,
                                            size_guidance={  # see below regarding this argument
                                                             event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                             event_accumulator.IMAGES: 4,
                                                             event_accumulator.AUDIO: 4,
                                                             event_accumulator.SCALARS: 0,
                                                             event_accumulator.HISTOGRAMS: 1,
                                            })

    ea.Reload()
    wanted_scalars = ['Train/loss_t',
                      'Train/loss_x',
                      'Train/acc',
                      'Train/lr',
                      'Test/gta_synthreid/rank1',
                      'Test/gta_synthreid/mAP']
    print("wanted scalars: ", wanted_scalars)
    print(ea.Tags())
    print()
    print(type(ea.Scalars('Test/gta_synthreid/mAP')[0].value), ea.Scalars('Test/gta_synthreid/mAP')[0].value)
    # sys.exit()

    for scalar in wanted_scalars:
        elements = ea.Scalars(scalar)
        print(elements)
        sys.exit()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--tfevents-files',
        type=str,
        nargs='+',
        required=True,
        help='Paths to the tfevents files of tensorboard, 1 file only for each desired experiment'
    )

    args = parser.parse_args()

    files = args.tfevents_files
    print("tfevents files: ", files)

    for file in files:
        get_data(file)


if __name__ == '__main__':
    main()
