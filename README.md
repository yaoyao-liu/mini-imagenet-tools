![mini-ImageNet Logo](https://github.com/y2l/mini-imagenet-tools/blob/master/mini-imagenet.png)
# Tools for mini-ImageNet Dataset

[![LICENSE](https://img.shields.io/github/license/y2l/mini-imagenet-tools.svg)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/miniimagenettools)](https://pypi.org/project/miniimagenettools/)
[![CodeFactor](https://www.codefactor.io/repository/github/yaoyao-liu/mini-imagenet-tools/badge)](https://www.codefactor.io/repository/github/y2l/mini-imagenet-tools)

This repo provides python source code for creating mini-ImageNet dataset from ImageNet and the utils for generating batches during training. This repo is related to our work on few-shot learning: [Meta-Transfer Learning](https://github.com/y2l/meta-transfer-learning-tensorflow).


### Summary

* [About mini-ImageNet](#about-mini-ImageNet)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Performance](#performance)
* [Acknowledgement](#acknowledgement)

### About mini-ImageNet

The mini-ImageNet dataset was proposed by [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) for few-shot learning evaluation. Its complexity is high due to the use of ImageNet images but requires fewer resources and infrastructure than running on the full [ImageNet dataset](https://arxiv.org/pdf/1409.0575.pdf). In total, there are 100 classes with 600 samples of 84×84 color images per class. These 100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test.

### Requirements

- Python 2.7 or 3.x
- numpy
- tqdm
- opencv-python
- Pillow

### Installation
Install via PyPI:
```bash
pip install miniimagenettools
```

Install via GitHub:
```bash
git clone https://github.com/yaoyao-liu/mini-imagenet-tools.git
```

### Usage 
First, you need to download the image source files from [ImageNet website](http://www.image-net.org/challenges/LSVRC/2012/). If you already have it, you may use it directly.
```
Filename: ILSVRC2012_img_train.tar
Size: 138 GB
MD5: 1d675b47d978889d74fa0da5fadfb00e
```
Then clone the repo:
```
git clone https://github.com:y2l/mini-imagenet-tools.git
cd mini-imagenet-tools
```
To generate mini-ImageNet dataset from tar file:
```bash
python mini_imagenet_generator.py --tar_dir [your_path_of_the_ILSVRC2012_img_train.tar]
```
To generate mini-ImageNet dataset from untarred folder:
```bash
python mini_imagenet_generator.py --imagenet_dir [your_path_of_imagenet_folder]
```
If you want to resize the images to the specified resolution:
```bash
python mini_imagenet_generator.py --tar_dir [your_path_of_the_ILSVRC2012_img_train.tar] --image_resize 100
```
P.S. In default settings, the images will be resized to 84 × 84. 

If you don't want to resize the images, you may set ```--image_resize 0```.

To use the ```MiniImageNetDataLoader``` class:
```python
from miniimagenettools.mini_imagenet_dataloader import MiniImageNetDataLoader

dataloader = MiniImageNetDataLoader(shot_num=5, way_num=5, episode_test_sample_num=15)

dataloader.generate_data_list(phase='train')
dataloader.generate_data_list(phase='val')
dataloader.generate_data_list(phase='test')

dataloader.load_list(phase='all')

for idx in range(total_train_step):
    episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
        dataloader.get_batch(phase='train', idx=idx)
    ...
```
### Performance
5-way classification accuracy (%)

|Method|1-shot|5-shot|
|---|---|---|
|[MAML](https://arxiv.org/pdf/1703.03400.pdf)| 48.70 ± 1.75| 63.11 ± 0.92|
|[ProtoNets](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)| 49.42 ± 0.78 |68.02 ± 0.66|
|[SNAIL](https://openreview.net/pdf?id=B1DmUzWAW)| 55.71 ± 0.99 |68.88 ± 0.92|
|[TADAM](https://arxiv.org/pdf/1805.10123.pdf)| 58.5 ± 0.3 |76.7 ± 0.3|
|[MTL](https://arxiv.org/pdf/1812.02391.pdf)| 61.2 ± 1.8 |75.5 ± 0.8|

### Download Processed Images 

[Download jpg files](https://drive.google.com/open?id=137M9jEv8nw0agovbUiEN_fPl_waJ2jIj) (Thanks for the contribution by [@vainaijr](https://github.com/vainaijr))

[Download tar files](https://meta-transfer-learning.yaoyao-liu.com/download/)


### Acknowledgement
[Model-Agnostic Meta-Learning](https://github.com/cbfinn/maml)

[Optimization as a Model for Few-Shot Learning](https://github.com/gitabcworld/FewShotLearning)
