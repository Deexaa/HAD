# [CVPR2026] HAD: Heterogeneity-Aware Distillation for Lifelong Heterogeneous Learning

This repository contains the code and supplementary material for our CVPR 2026 paper.

![framework](figures/framework.png)

## Requirements

* PyTorch
* torchvision
* numpy
* opencv-python
* pillow
* scipy
* scikit-image
* six
* tqdm
* matplotlib

## Datasets

Currently, we provide the implementation for the _NYUv2_ dataset. The code for the remaining datasets is being organized and will be released as soon as possible.

Please arrange the dataset as follows:

```text
/data/dataset/nyuv2/
├── train/
│   ├── image/
│   ├── label/
│   ├── depth/
│   └── normal/
└── val/
    ├── image/
    ├── label/
    ├── depth/
    └── normal/
```

The dataset path is set in `datasetting/continual_dataset.py`.

## Running scripts

```bash
$ python main.py
```

## Acknowledgment

Our project references the codes in the following repos.

* [MTSAM](https://github.com/XuehaoWangFi/MTSAM)
* [MAMMOTH](https://github.com/aimagelab/mammoth)
* [SGP](https://github.com/sahagobinda/SGP)
* [SPG](https://github.com/UIC-Liu-Lab/spg)

## Citation

If you find our work useful in your research, please consider citing our paper.

```bibtex
@inproceedings{had,
  title={HAD: Heterogeneity-Aware Distillation for Lifelong Heterogeneous Learning},
  author={Zhang, Xuerui and Wang, Xuehao and Zhuang, Zhan and Zhao, Linglan and Li, Ziyue and Zhang, Xinmin and Song, Zhihuan and Zhang, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10863--10873},
  year={2026}
}
```
