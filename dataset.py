import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.io import loadmat
import os
from PIL import Image
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd

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
        transfTar = transforms.ToTensor()
        std=[0.229, 0.224, 0.225]
        mean=[0.485, 0.456, 0.406]
        transfIm = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean,std)
        ])
        inputImage = transfIm(Image.open(self.rootDirImg + inputName).convert('RGB'))

        targetImage = transfTar(Image.open(self.rootDirGt + targetName).convert('L'))
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

def grayTrans(img):
    img = img.data.cpu().numpy()[0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

class TrainDataset(Dataset):
    def __init__(self, fileNames, rootDir):        
        self.rootDir = rootDir
        self.transform = transforms.ToTensor()
        self.targetTransform = transforms.ToTensor()
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # input and target images
        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0])
        targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1])

        # process the images
        inputImage = Image.open(inputName).convert('RGB')
        if self.transform is not None:
            inputImage = self.transform(inputImage)

        targetImage = Image.open(targetName).convert('L')
        if self.targetTransform is not None:
            targetImage = self.targetTransform(targetImage)
            targetImage = (targetImage>0.41).float()
        return inputImage, targetImage
