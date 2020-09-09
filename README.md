![mini-ImageNet Logo](https://github.com/y2l/mini-imagenet-tools/blob/master/mini-imagenet.png)
# Tools for mini-ImageNet Dataset

[![LICENSE](https://img.shields.io/github/license/y2l/mini-imagenet-tools.svg)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/miniimagenettools)](https://pypi.org/project/miniimagenettools/)
[![Downloads](https://pepy.tech/badge/miniimagenettools)](https://pepy.tech/project/miniimagenettools)
![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/yaoyao-liu/mini-imagenet-tools)

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

Please note that the split files in `csv_files` folder are created by [Ravi and Larochelle](https://openreview.net/pdf?id=rJY0-Kcll) ([GitHub link](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet)). [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) didn't include their split files for mini-ImageNet when they first released their paper, so [Ravi and Larochelle](https://openreview.net/pdf?id=rJY0-Kcll) created their own splits. Additional split files are provided [here](https://github.com/yaoyao-liu/mini-imagenet-tools/tree/master/mini_imagenet_split).

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
Some people report the ImageNet website is not working. Here is an [alternative download link](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2). Please carefully read the terms for ImageNet before you download it.
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

Check the SOTA results for mini-ImageNet on [this page](https://few-shot.yyliu.net/miniimagenet.html).

### Download Processed Images 

[Download jpg files](https://drive.google.com/open?id=137M9jEv8nw0agovbUiEN_fPl_waJ2jIj) (Thanks for the contribution by [@vainaijr](https://github.com/vainaijr))

[Download tar files](https://mtl.yyliu.net/download/)


### Acknowledgement
[Model-Agnostic Meta-Learning](https://github.com/cbfinn/maml)

[Optimization as a Model for Few-Shot Learning](https://github.com/gitabcworld/FewShotLearning)

[Meta-Learning for Semi-Supervised Few-Shot Classification](https://github.com/renmengye/few-shot-ssl-public)

[@ChristopherDaw](https://github.com/ChristopherDaw)
