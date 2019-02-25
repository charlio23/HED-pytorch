# Pytorch HED reimplementation

This piece of code consists on a reimplementation of [Holistically-Nested Edge Detection](https://arxiv.org/abs/1504.06375) using Python 3.6 and Pytorch 1.0.1.

**This code is incomplete** but it has several main features which can be used.

The main objective is to reproduce the algorithm as it is done in the official implementation using only Python.

### Contents

This repository contain two executable files:

- [image_preprocesing.py](https://github.com/charlio23/HED-pytorch/blob/master/image_preprocesing.py): Data preprocessing and augmentation
- [main.py](https://github.com/charlio23/HED-pytorch/blob/master/main.py): Training

### Data preprocessing and augmentation

The script reads from the BSDS500 dataset (.mat files) and converts its content to .png files. It generates the ground truth images at it is specified in the official implementation. Later, it performs the data augmentation: flip, rotate and crop the largest rectangle of the images to obtain a factor of 32 augmented data. Finally the images are resized to 400x400.

The results are stored by default in a new directory -> BSDS500_AUGMENTED

You can download the BSDS500 dataset [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz).

```
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
tar -xvf BSR_bsds500.tgz
mv BSR/BSDS500/ ./
rm -rf BSR/
python image_preprocesing.py
```

(uncomment the preprocess step before executing)

### Training

Once you have your augmented data, you can try training the algorithm.

First you need to download the VGG16 pretrained model.

```
mkdir model
cd model
wget https://download.pytorch.org/models/vgg16-397923af.pth
mv vgg16-397923af.pth vgg16.pth
python main.py
```

### TODO

- Draw precision-recall curves and monitor performance
- Try parameter tuning

## References

- [Holistically-nested edge detection](https://arxiv.org/abs/1504.06375) S. Xie and Z. Tu, in Proc. ICCV, 2015, pp. 1395–1403.
- [Contour Detection and Hierarchical Image Segmentation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf) P. Arbelaez, M. Maire, C. Fowlkes and J. Malik. IEEE TPAMI, Vol. 33, No. 5, pp. 898-916, May 2011.
