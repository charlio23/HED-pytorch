import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.io import loadmat
import os
from PIL import Image
import numpy as np
from torch.autograd import Variable

class TrainDataset(Dataset):
    def __init__(self, rootDirImg, rootDirGt):
        self.rootDirImg = rootDirImg
        self.rootDirGt = rootDirGt
        self.listData = [sorted(os.listdir(rootDirImg)),sorted(os.listdir(rootDirGt))]
        

    def __len__(self):
        return len(self.listData[1])
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[0][i]
        targetName = self.listData[1][i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))

        targetImage = loadmat(self.rootDirGt + targetName)['groundTruth']
        targetImage = np.int32(targetImage[0][0][0][0][1])
        targetImage = Variable(torch.from_numpy(targetImage))
        return inputImage, targetImage
