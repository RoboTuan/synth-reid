import argparse
import os


def main(datafolder, newfolder):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', type=str, default='/mnt/data2/defonte_data/PersonReid_datasets/market1501')
    parser.add_argument('--newfolder', type=str, default='/mnt/data2/defonte_data/PersonReid_datasets/fid/market_flat')
    args = parser.parse_args()

    datafolder = args.datafolder
    newfolder = args.newfolder

    main(datafolder, newfolder)
