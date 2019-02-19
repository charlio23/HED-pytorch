import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.io import loadmat
import os
from PIL import Image
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

class BSDS(Dataset):
    def __init__(self, rootDirImg, rootDirGt, processed=True):
        self.rootDirImg = rootDirImg
        self.rootDirGt = rootDirGt
        self.listData = [sorted(os.listdir(rootDirImg)),sorted(os.listdir(rootDirGt))]
        self.processed = processed

    def __len__(self):
        return len(self.listData[1])
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[0][i]
        targetName = self.listData[1][i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))

        targetImage = transf(Image.open(self.rootDirImg + inputName).convert('L'))
        return inputImage, targetImage

    def preprocess(self):
        if self.processed:
            return

        for targetName in self.listData[1]:
            targetFile = loadmat(self.rootDirGt + targetName)['groundTruth'][0]
            s = (len(targetFile[0][0][0][1]), len(targetFile[0][0][0][1][0]))
            result = np.zeros(s)
            for target in targetFile:
                element = target[0][0][1]
                result = np.add(result, element)
            appl = np.vectorize(lambda x: 1 if x >= 3 else 0)
            result = appl(result)
            # save result as png image
            result = (result*255.0).astype(np.uint8)
            img = Image.fromarray(result, 'L')
            targetSaveName = targetName.replace(".mat", ".png")
            img.save(self.rootDirGt + targetSaveName)
        # erase all .mat files
        os.system("rm " + self.rootDirGt + "*.mat")
        # update dataset
        self.listData = [sorted(os.listdir(self.rootDirImg)),sorted(os.listdir(self.rootDirGt))]
        self.processed = True
