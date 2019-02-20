from dataset import BSDS
from model import initialize_hed
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from PIL import Image
from torch import sigmoid
from torch.nn.functional import binary_cross_entropy
from torch.autograd import Variable
import time

def grayTrans(img):
    img = img.data.cpu().numpy()[0][0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

def colorTrans(img):
    img = img.data.cpu().numpy()[0]
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'RGB')
    return img

print("Importing datasets...")

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

print("Initializating network...")

modelPath = "model/vgg16.pth"

nnet = initialize_hed(modelPath)
nnet.cuda()

train = DataLoader(trainDS, shuffle=True)
val = DataLoader(valDS, shuffle=True)
test = DataLoader(testDS, shuffle=False)

print("Defining hyperparameters...")

### HYPER-PARAMETERS
learningRate = 1e-6
momentum = 0.9
miniBatchSize = 10
lossWeight = 1
initializationNestedFilters = 0
initializationFusionWeights = 1/5
weightDecay = 0.0002
trainingIterations = 10000
###

def bce2d(input, target):    
        
        target_trans = target.clone()
        pos_index = (target >0.5)
        neg_index = (target <0.5)        
        weight = torch.Tensor(input.size()).fill_(0)
        pos_num = pos_index.sum().item()
        neg_num = neg_index.sum().item()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num
        weight = weight.cuda()

        loss = binary_cross_entropy(input, target, weight)
        return loss




optimizer = optim.SGD(nnet.parameters(), lr=learningRate, momentum=momentum, weight_decay=weightDecay)

print("Training started")

epochs = int(np.ceil(trainingIterations/(len(train) + len(val))))
i = 0
dispInterval = 100
lossAcc = 0.0
epoch_line = []
loss_line = []
for epoch in range(epochs):
    print("Epoch: " + str(epoch + 1))
    for j, data in enumerate(train, 0):
        image, target = data
        image, target = Variable(image).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        side1, side2, side3, side4, side5, fuse = nnet(image)
        loss1 = bce2d(side1, target)
        loss2 = bce2d(side2, target)
        loss3 = bce2d(side3, target)
        loss4 = bce2d(side4, target)
        loss5 = bce2d(side5, target)
        loss6 = binary_cross_entropy(fuse, target)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        loss.backward()
        optimizer.step()
        lossAcc += loss.item()
        
        if (i+1) % dispInterval == 0:
                    timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    lossDisp = lossAcc/dispInterval
                    epoch_line.append(i)
                    loss_line.append(lossDisp)
                    print("%s epoch: %d iter:%d loss:%.6f"%(timestr, epoch+1, i+1, lossDisp))
                    lossAcc = 0.0
        i += 1
    # transform to grayscale images
    side1 = grayTrans(side1)
    side2 = grayTrans(side2)
    side3 = grayTrans(side3)
    side4 = grayTrans(side4)
    side5 = grayTrans(side5)
    fuse = grayTrans(fuse)
    avg = grayTrans((side1 + side2 + side3 + side4 + side5 + fuse)/6)
    tar = grayTrans(target)
    
    side1.save('images/sample_1.png')
    side2.save('images/sample_2.png')
    side3.save('images/sample_3.png')
    side4.save('images/sample_4.png')
    side5.save('images/sample_5.png')
    fuse.save('images/sample_6.png')
    avg.save('images/sample_7.png')
    tar.save('images/sample_T.png')
    torch.save(nnet.state_dict(), 'HED.pth')

plt.plot(epoch_line,loss_line)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig("images/loss.png")
