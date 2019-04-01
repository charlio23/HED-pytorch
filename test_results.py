import os
import torch
from dataset import BSDS_TEST
from model import HED
from torch.autograd import Variable
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

def grayTrans(img):
    img = img.data.cpu().numpy()[0][0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

print("Loading train dataset...")

rootDirImgTest = "BSDS500/data/images/test/"
testOutput = "test/output/"

testDS = BSDS_TEST(rootDirImgTest)
test = DataLoader(testDS, shuffle=False)

os.makedirs(testOutput, exist_ok=True)

print("Loading trained network...")

networkPath = "HED2.pth"

nnet = HED().cuda()
dic = torch.load(networkPath)
dicli = list(dic.keys())
new = {}
j = 0
for k in nnet.state_dict():
    new[k] = dic[dicli[j]]
    j += 1
nnet.load_state_dict(new)

print("Generating test results...")

for j, data in enumerate(tqdm(test), 0):
    image, imgName = data
    image = Variable(image).cuda()
    sideOuts = nnet(image)
    fuse = grayTrans(sideOuts[-1])
    fuse.save(testOutput + imgName[0])