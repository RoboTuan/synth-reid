import sys
sys.path.append(".")
import torch
from torch.utils.data import Dataset, DataLoader
from torchreid.data.transforms import build_transforms
from torchreid.utils import read_image, load_pretrained_weights, set_random_seed
from torchvision.transforms import Compose, Resize
from PIL import Image
import torchvision
from torchreid.models import Generator
import argparse
import os


class GTA_flatten(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data = []
        self.transform = transform
        for img_path in os.listdir(self.root_dir):
            if img_path[-5:] == '.jpeg' or img_path[-4:] == '.png':
                self.data.append(img_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img_path = os.path.join(self.root_dir, img_path)
        img = read_image(img_path)

        if self.transform:
            img = self.transform(img)

        item = {
            'img': img,
            'pth': img_path
        }

        return item


def main(datafolder, newfolder, weights, seed=10):
    set_random_seed(seed)

    if not os.path.isdir(newfolder):
        print(f"Creating folder {newfolder}...")
        os.makedirs(newfolder)

    batch = 100
    transforms, _ = build_transforms(
        width=128,
        height=256,
        transforms=[],
        norm_mean=[0.5, 0.5, 0.5],
        norm_std=[0.5, 0.5, 0.5]
    )
    save_transforms = Compose([Resize((128, 64), interpolation=Image.BILINEAR)])

    dataset = GTA_flatten(datafolder, transforms)
    loader = DataLoader(dataset, batch, shuffle=False, drop_last=False)
    generator_S2R = Generator()
    load_pretrained_weights(generator_S2R, weights)
    generator_S2R.cuda()
    # generator_S2R.eval()
    for p in generator_S2R.parameters():
        p.requires_grad = False

    with torch.no_grad():
        for data in loader:
            imgs, pths = data['img'], data['pth']
            imgs = imgs.cuda()
            new_imgs = generator_S2R(imgs)
            new_imgs = (new_imgs.data + 1) / 2.0

            for i, img in enumerate(new_imgs):
                img_pth = pths[i].split("/")[-1]
                dest = os.path.join(newfolder, img_pth)
                img = save_transforms(img)
                torchvision.utils.save_image(img, dest)
            # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', type=str, default='/mnt/data2/defonte_data/PersonReid_datasets/fid/gta_flat')
    parser.add_argument('--newfolder', type=str, default='/mnt/data2/defonte_data/PersonReid_datasets/fid/cuhk03_gta_flat_tra_sim')
    parser.add_argument('--weights', type=str, default=str('/home/defonte/PersonReid/synth-reid_adv/log/' +
                        'cuhk03_adv_nce_resnet_joint_test_11_bottle/generator/model.pth.tar-60'))
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    datafolder = args.datafolder
    newfolder = args.newfolder
    weights = args.weights
    seed = args.seed

    main(datafolder, newfolder, weights, seed)
