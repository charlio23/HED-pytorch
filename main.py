from dataset import TrainDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def grayTrans(img):
    img = img.data.cpu().numpy()[0]*255/(torch.max(img).item())
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

def edgeTrans(img):
    img = bwdist(1 - img.numpy()[0])
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

rootDirImg = "BSDS500/data/images/train/"
rootDirGt = "BSDS500/data/groundTruth/train/"

train = DataLoader(TrainDataset(rootDirImg,rootDirGt),shuffle=True)

sample, target = iter(train).next()
print(sample)
fig = plt.figure(figsize=(8,8))
plt.imshow(np.transpose(sample[0].numpy(), (1, 2, 0)))
plt.imshow(grayTrans(target))
plt.show()