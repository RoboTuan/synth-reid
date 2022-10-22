# Synth-ReID
Synthetic-to-Real Domain Transfer with Joint Image Translation and Discriminative Learning for Pedestrian Re-Identification.

This is my master thesis project for the M.Sc. degree in Data Science and Engineering at Politecnico di torino. It extends my previous [internship](https://github.com/RoboTuan/GTASynthReid) as both works were completed at the Links Foundation. The code is based on [this](https://github.com/KaiyangZhou/deep-person-reid) framework.
The objective was to generalize from our dataset [GTASynthReid](https://github.com/RoboTuan/GTASynthReid) to real-world person re-identification data using generative methods. We employed the *contrastive unpaired translation* (CUT) framework for image translation with joint discriminative learning for synth-to-real adaptation.

In the following picture there are some examples of translated pedetrians to three different real-world datasets (Market, Duke, CUHK03).

![translations](https://github.com/RoboTuan/synth-reid/blob/main/images/Translation_examples.svg)


# Architecture
Our network is based on CUT and includes the domain mapping, relationship preservation and discriminative modules. It has a single GAN since it does not need cycle consistency. Discriminative learning is embedded in the image translation process

## Domain mapping
The synth-to-real domain mapper is a GAN with a encoder-decoder generator (see picture) and a PatchGan discriminator. We exploted the L2 loss for the adversarial training. We also engineered an adversaril feature matching loss asking for similar features at certain discriminator's layer. From the generator's perspective, we want this to hapeen between translated and target images. From the discriminator's perspective, instead, between source and target images.

![encoder-decoder](https://github.com/RoboTuan/synth-reid/blob/main/images/encoder_decoder.svg)

However, this is not enough to produce images that are more similar to the target ones as one can see from the following image (left is source, right is translation). We need to better constrain and guide the generative process.

![encoder-decoder](https://github.com/RoboTuan/synth-reid/blob/main/images/Gan_only.png)


## Relationship preservation
We adopted CUT to guide the generative process instead of solutions that employ some kind of cycle consistency, avoiding bijective mappings. Intuitively, this method asks corresponding patches of the input and output images to be similar after the translation (see image below). The patches are simply the output feature vectors along the channel dimension of a given encoder's layer *l* (receptive field arithmetic). These feature vectors are then further processed by smaller multilayer perceptrons (*H* in image) Then we just perform a patch classification. This process is repeated along different layers of the generator's encoder. 

![encoder-decoder](https://github.com/RoboTuan/synth-reid/blob/main/images/Patches.svg)


## Discriminative learning
We still need to explain how we integrated the re-identificatoin task with image translation. One option would be to train a network for re-identification on the translated images after having translated the whle source data. However, this would add another training stage and thus slow the entire process. We instead processed with a Resnet the features coming from the last layer of the generator's encoder. Theese feature are responsible for the translation and hold some information about the source identities. They are not yet specialized to discriminate among different pedestrians. For this reason, we simply trained the Resnet with cross-entropy loss for classification on the source idetities using as input features with target-related information. 

![encoder-decoder](https://github.com/RoboTuan/synth-reid/blob/main/images/ReID_net.svg)


# Results
spiegare i 2 tipi di risultati e come viene usata la rete perf infrerenza
## Quantitative results
## Qualitatiive results


# References

    @article{cut,
      title={Contrastive Learning for Unpaired Image-to-Image Translation},
      author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
      booktitle={European Conference on Computer Vision},
      year={2020}
    }

    @article{torchreid,
      title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch},
      author={Zhou, Kaiyang and Xiang, Tao},
      journal={arXiv preprint arXiv:1910.10093},
      year={2019}
    }
    
    @inproceedings{zhou2019osnet,
      title={Omni-Scale Feature Learning for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      booktitle={ICCV},
      year={2019}
    }

    @article{zhou2021osnet,
      title={Learning Generalisable Omni-Scale Representations for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      journal={TPAMI},
      year={2021}
    }
