import sys
sys.path.append(".")
import argparse
import os
import torchreid.data
from torchreid.utils import set_random_seed
import torchvision


def main(datafolder, source, newfolder, seed):
    set_random_seed(seed)

    if not os.path.isdir(newfolder):
        print(f"Creating folder {newfolder}...")
        os.makedirs(newfolder)

    datamanager = torchreid.data.ImageDataManager(
        root=datafolder,
        sources=source,
        height=128,
        width=64,
        batch_size_train=8,
        batch_size_test=100,
        combineall=True,
        seed=seed,
        norm_mean=[0.5] * 3,
        norm_std=[0.5] * 3,
        n_samples=50  # taking only at max 50 images per identity for GTA_synthReid
    )

    for data in datamanager.train_loader:
        imgs = data['img']
        imgs = (imgs.data + 1) / 2.0
        paths = data['impath']

        for i, img in enumerate(imgs):
            img_pth = paths[i].split("/")[-1]
            dest = os.path.join(newfolder, img_pth)
            torchvision.utils.save_image(img, dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', type=str, default='/mnt/data2/defonte_data/PersonReid_datasets/')
    parser.add_argument('--source', type=str, default='cuhk03')
    parser.add_argument('--newfolder', type=str, default='/mnt/data2/defonte_data/PersonReid_datasets/fid/cuhk03_flat')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    datafolder = args.datafolder
    newfolder = args.newfolder
    source = args.source
    seed = args.seed

    main(datafolder, source, newfolder, seed)
