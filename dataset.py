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
import cv2
from pycocotools.coco import COCO as COCO_
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt

def drawEdges(segment, width=1):
    contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.drawContours((segment*0.0).astype(np.uint8), contours, -1, 255, width).get()
    return edges

def drawSkeleton(segment):
    return skeletonize(segment)

class COCO(Dataset):
    def __init__(self, annotationPath, rootDirImg, offline=False):
        self.annotationPath = annotationPath
        self.coco = COCO_(annotationPath)
        self.rootDirImg = rootDirImg
        self.offline = offline
    def __len__(self):
        catIds = self.coco.getCatIds(catNms=['person'])
        imgID = self.coco.getImgIds(catIds=catIds)
        return len(imgID)
                
    def __getitem__(self, i):
        # input and target images
        catIds = self.coco.getCatIds(catNms=['person'])
        imgID = self.coco.getImgIds(catIds=catIds)[i]
        annIds = self.coco.getAnnIds(imgIds=imgID)
        image = self.coco.loadImgs(imgID)[0]
        # process the images
        transf = transforms.ToTensor()
        inputImage = torch.tensor([])
        if not self.offline:
            inputImage = transf(io.imread(image['coco_url']))
        else:
            inputName = image["file_name"]
            inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))
        tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434

        inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)
        annotations = self.coco.loadAnns(annIds)
        annList = [self.coco.annToMask(annotation) for annotation in annotations]
        if len(annList) == 0:
            return [], []
        merge = np.vectorize(lambda x, y: np.uint8(255) if (x > 0.5 or y > 0.5) else np.uint8(0))
        edges = drawEdges(annList[0])
        for segment in annList[1:]:
            new_edges = drawEdges(segment)
            edges = merge(edges,new_edges)
        targetImage = (torch.from_numpy(edges)>0.5).unsqueeze_(0).float()
        return inputImage, targetImage

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
        targetImage = transf(Image.open(self.rootDirGt + targetName).convert('L'))
        targetImage = (targetImage>0.41).float()
        return inputImage, targetImage

    def preprocess(self):
        if self.processed:
            return

        for targetName in self.listData[1]:
            targetFile = loadmat(self.rootDirGt + targetName)['groundTruth'][0]
            s = (len(targetFile[0][0][0][1]), len(targetFile[0][0][0][1][0]))
            result = np.zeros(s)
            num = len(targetFile)
            for target in targetFile:
                element = target[0][0][1]
                result = np.add(result, element)
            
            result = result/num
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

class BSDS_TEST(Dataset):
    def __init__(self, rootDirImg):
        self.rootDirImg = rootDirImg
        self.listData = sorted(os.listdir(rootDirImg))

    def __len__(self):
        return len(self.listData)
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))
        inputName = inputName.split(".jpg")[0] + ".png"
        tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434
        inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)
        return inputImage, inputName

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
        inputImage = self.transform(inputImage)

        tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434

        inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)

        targetImage = Image.open(targetName).convert('L')
        targetImage = self.targetTransform(targetImage)
        targetImage = (targetImage>0.41).float()
        return inputImage, targetImage