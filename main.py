from dataset import BSDS
from model import HED
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def grayTrans(img):
    img = img.data.cpu().numpy()[0]*255
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

def colorTrans(img):
    img = img.data.cpu().numpy()[0]
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'RGB')
    return img

rootDirImgTrain = "BSDS500/data/images/train/"
rootDirGtTrain = "BSDS500/data/groundTruth/train/"
rootDirImgVal = "BSDS500/data/images/val/"
rootDirGtVal = "BSDS500/data/groundTruth/val/"
rootDirImgTest = "BSDS500/data/images/test/"
rootDirGtTest = "BSDS500/data/groundTruth/test/"

preprocessed = False # Set this to False if you want to preprocess the data
trainDS = BSDS(rootDirImgTrain, rootDirGtTrain, preprocessed)
valDS = BSDS(rootDirImgVal, rootDirGtVal, preprocessed)
testDS = BSDS(rootDirImgTest, rootDirGtTest, preprocessed)

# Uncoment if you want to do preprocessing (.mat -> .png)
#trainDS.preprocess()
#valDS.preprocess()
#testDS.preprocess()

modelPath = "model/vgg16.pth"


