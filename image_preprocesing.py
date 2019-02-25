from dataset import BSDS
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from tqdm import tqdm

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

## SECOND STEP: DEFINDE LOADING AND SAVING FUNCTIONS

print("Defining loading, saving and operand functions for images...")

def load_images(path):
    image_list = sorted(os.listdir(path))
    image_hash = {}
    for image_name in image_list:
        im = Image.open(path + image_name)
        image_hash[image_name] = im

    return image_hash

def rotate_if_needed(image_hash):
    for k, v in image_hash.items():
        if (v.size[0] < v.size[1]):
            v = v.rotate(90,expand=True)
            image_hash[k] = v

    return image_hash

def add_flipped_images(image_hash):
    print("Adding flipped images...")
    new_image_hash = {}
    for k, v in tqdm(image_hash.items()):
        k_new = k.split(".")[0]
        ext = ".jpg" #feel free to change the extension
        new_image_hash[k_new + "_0" + ext] = v
        new_image_hash[k_new + "_1" + ext] = v.transpose(Image.FLIP_TOP_BOTTOM)

    return new_image_hash

def rotate_and_crop(im, angle):
    rotated = im.rotate(angle, Image.BICUBIC, True)
    aspect_ratio = float(im.size[0]) / im.size[1]
    rotated_aspect_ratio = float(rotated.size[0]) / rotated.size[1]
    alp = math.fabs(angle) * math.pi / 180

    if aspect_ratio < 1:
        total_height = float(im.size[0]) / rotated_aspect_ratio
    else:
        total_height = float(im.size[1])

    h = abs(total_height / (aspect_ratio * abs(math.sin(alp)) + abs(math.cos(alp))))
    w = abs(h * aspect_ratio)

    a = rotated.size[0]*0.5
    b = rotated.size[1]*0.5

    return rotated.crop((a - w*0.5, b - h*0.5, a + w*0.5, b + h*0.5))

def add_rotated_crop_images(image_hash, angle_list):
    print("Adding rotated images...")
    new_image_hash = {}

    for k, v in tqdm(image_hash.items()):
        k_new = k.split(".")[0]
        ext = ".jpg" #feel free to change the extension
        for angle in angle_list:
            new_image_hash[k_new + "_" + str(int(angle)) + ext] = rotate_and_crop(v, angle)

    return new_image_hash

def resize_images(image_hash, dim):
    print("Resizing images...")
    for k, v in tqdm(image_hash.items()):
        image_hash[k] = v.resize(dim)

    return image_hash

def save_images(path,image_hash):
    os.makedirs(path, exist_ok=True)
    for k, v in image_hash.items():
        v.save(path + k)
        print("Image " + k +  " saved at " + path)

###########################

## THIRD STEP: CREATE DATA AUGMENTATION

print("Augmenting data...")

destDirImgTrain = "BSDS500_AUGMENTED/data/images/train/"
destDirGtTrain = "BSDS500_AUGMENTED/data/groundTruth/train/"
destDirImgVal = "BSDS500_AUGMENTED/data/images/val/"
destDirGtVal = "BSDS500_AUGMENTED/data/groundTruth/val/"
destDirImgTest = "BSDS500_AUGMENTED/data/images/test/"
destDirGtTest = "BSDS500_AUGMENTED/data/groundTruth/test/"

## List of directories for source and destination files
source_dest_files = [(rootDirImgTrain, destDirImgTrain),
                    (rootDirGtTrain, destDirGtTrain),
                    (rootDirImgVal, destDirImgVal),
                    (rootDirGtVal, destDirGtVal),
                    (rootDirImgTest, destDirImgTest),
                    (rootDirGtTest, destDirGtTest)]

new_size = (400, 400)
num_angles = 17

angle_list = np.linspace(0,360,num_angles)[:-1]

for source, dest in source_dest_files:
    print("Reading from: " + source)
    print("Saving at: " + dest)
    image_hash = resize_images(
                    add_rotated_crop_images(
                        add_flipped_images(
                            rotate_if_needed(
                                load_images(source)))
                    ,angle_list)
                ,new_size)
    save_images(dest,image_hash)

###########################
