from dataset import BSDS
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

print("Importing datasets...")

rootDirImgTrain = "BSDS500/data/images/train/"
rootDirGtTrain = "BSDS500/data/groundTruth/train/"
rootDirImgVal = "BSDS500/data/images/val/"
rootDirGtVal = "BSDS500/data/groundTruth/val/"
rootDirImgTest = "BSDS500/data/images/test/"
rootDirGtTest = "BSDS500/data/groundTruth/test/"

trainDS = BSDS(rootDirImgTrain, rootDirGtTrain, processed=True)
valDS = BSDS(rootDirImgVal, rootDirGtVal, processed=True)
testDS = BSDS(rootDirImgTest, rootDirGtTest, processed=True)

############################

## FIRST STEP: FROM .mat TO .png
# Uncoment if you want to do this preprocessing (.mat -> .png)
#trainDS.preprocess()
#valDS.preprocess()
#testDS.preprocess()

###########################

## SECOND STEP: DATA AUGMENTATION

def rotate_if_needed(old_path,new_path):
	os.makedirs(new_path, exist_ok=True)
	li = os.listdir(old_path)
	for l in li:
		im = Image.open(old_path + l)
		if (im.size[0] == 321):
			print("rotate and resize!")
			im = im.rotate(90,expand=True)

		im.save(new_path + l)

def flip_images(path):
	li = os.listdir(path)
	for l in li:
		im = Image.open(path + l)
		im = im.transpose(Image.FLIP_TOP_BOTTOM)
		im.save(path + l)

rotate_if_needed(rootDirImgTrain,"BSDS500_AUGMENTED/train/")
#flip_images("BSDS500_AUGMENTED/train/")