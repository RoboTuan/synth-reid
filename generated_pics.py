import torchreid
import torch.nn as nn
import torch
from torchreid.models import Generator
import torchvision
from torchreid.utils import set_random_seed, load_pretrained_weights, resume_from_checkpoint
import sys

seed = 10
generator_path = './log/adv_nce_triplet_joint_val_12/generator/model.pth.tar-60'
dest_folder = './generated_imgs/'
set_random_seed(10)

datamanager = torchreid.data.ImageDataManager(
    root='/mnt/data2/defonte_data/PersonReid_datasets/',
    sources='gta_synthreid',
    targets='gta_synthreid',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=32,
    # transforms=None,
    # transforms=['random_flip', 'random_crop_translate'],
    val=False,
    adversarial=False,
    combineall=False,
    seed=seed,
    # workers=4,
    norm_mean=[0.5] * 3,
    norm_std=[0.5] * 3,
    n_samples=20  # taking only at max 20 images per identity for GTA_synthReid
)

generator_S2R = Generator()
load_pretrained_weights(generator_S2R, generator_path)
generator_S2R.cuda()
generator_S2R.eval()
for p in generator_S2R.parameters():
    p.requires_grad = False

for data in datamanager.train_loader:
    imgs = data['img']
    paths = data['impath']
    imgs = imgs.cuda()
    new_imgs = generator_S2R(imgs)
    imgs = (imgs.data + 1) / 2.0
    new_imgs = (new_imgs.data + 1) / 2.0
    for j, img_pair in enumerate(zip(imgs, new_imgs)):
        img = torch.unsqueeze(img_pair[0], 0)
        new_img = torch.unsqueeze(img_pair[1], 0)
        img_pair = torch.cat((img, new_img), dim=0)
        img_pth = dest_folder + paths[j].split("/")[-1]
        torchvision.utils.save_image(img_pair, img_pth, nrow=2)
