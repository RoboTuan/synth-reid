Synth-ReID
===========
Synthetic-to-Real Domain Transfer with Joint Image Translation and Discriminative Learning for Pedestrian Re-Identification.

This is the master thesis project for the Master's Degree in Data Science and Engineering at Politecnico di torino. It completes the previous [internship](https://github.com/RoboTuan/GTASynthReid) as both works were completed at the Links Foundation. The code is based on [this](https://github.com/KaiyangZhou/deep-person-reid) framework.
The objective was to generalize from our dataset [GTASynthReid](https://github.com/RoboTuan/GTASynthReid) to real-world person re-identification data using a generative framework. We employed the *contrastive unpaired translation* framework for image translation with joint discriminative learning for synth-to-real adaptation.

In the following picture there are some examples of translated pedetrians to three different real-world datasets.

![translations](https://github.com/RoboTuan/synth-reid/blob/main/images/Translation_examples.svg)

References
---------

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
