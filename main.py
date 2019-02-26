from dataset import BSDS, TrainDataset
from model import initialize_hed
from torch.utils.data import DataLoader, ConcatDataset
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
from itertools import chain
from tqdm import tqdm
from torch.optim import lr_scheduler
from collections import defaultdict

def grayTrans(img):
    img = img.data.cpu().numpy()[0][0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

print("Importing datasets...")

rootDirImgTrain = "BSDS500_AUGMENTED/data/images/train/"
rootDirGtTrain = "BSDS500_AUGMENTED/data/groundTruth/train/"
rootDirImgVal = "BSDS500_AUGMENTED/data/images/val/"
rootDirGtVal = "BSDS500_AUGMENTED/data/groundTruth/val/"
rootDirImgTest = "BSDS500_AUGMENTED/data/images/test/"
rootDirGtTest = "BSDS500_AUGMENTED/data/groundTruth/test/"

preprocessed = False # Set this to False if you want to preprocess the data
trainDS = BSDS(rootDirImgTrain, rootDirGtTrain, preprocessed)
valDS = BSDS(rootDirImgVal, rootDirGtVal, preprocessed)
trainDS = ConcatDataset([trainDS,valDS])

#trainDS = TrainDataset("HED-BSDS/train_pair.lst","HED-BSDS/")
#testDS = BSDS(rootDirImgTest, rootDirGtTest, preprocessed)

# Uncoment if you want to do preprocessing (.mat -> .png)
#trainDS.preprocess()
#valDS.preprocess()
#testDS.preprocess()

print("Initializing network...")

modelPath = "model/vgg16.pth"

nnet = torch.nn.DataParallel(initialize_hed(modelPath))
nnet.cuda()

train = DataLoader(trainDS, shuffle=True, batch_size=1, num_workers=4)

#test = DataLoader(testDS, shuffle=False)

print("Defining hyperparameters...")

### HYPER-PARAMETERS
learningRate = 1e-6
momentum = 0.9
miniBatchSize = 10
lossWeight = 1
initializationNestedFilters = 0
initializationFusionWeights = 1/5
weightDecay = 0.0002
###

def balanced_cross_entropy(input, target):            
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

    # Optimizer settings.
net_parameters_id = defaultdict(list)
for name, param in nnet.named_parameters():
    if name in ['module.conv1.0.weight', 'module.conv1.2.weight',
                'module.conv2.0.weight', 'module.conv2.1.weight',
                'module.conv3.1.weight', 'module.conv3.3.weight', 'module.conv3.5.weight',
                'module.conv4.1.weight', 'module.conv4.3.weight', 'module.conv4.5.weight']:
        print('{:26} lr:    1 decay:1'.format(name)); net_parameters_id['conv1-4.weight'].append(param)
    elif name in ['module.conv1.0.bias', 'module.conv1.2.bias',
                'module.conv2.0.bias', 'module.conv2.1.bias',
                'module.conv3.1.bias', 'module.conv3.3.bias', 'module.conv3.5.bias',
                'module.conv4.1.bias', 'module.conv4.3.bias', 'module.conv4.5.bias']:
        print('{:26} lr:    2 decay:0'.format(name)); net_parameters_id['conv1-4.bias'].append(param)
    elif name in ['module.conv5.1.weight', 'module.conv5.3.weight', 'module.conv5.5.weight']:
        print('{:26} lr:  100 decay:1'.format(name)); net_parameters_id['conv5.weight'].append(param)
    elif name in ['module.conv5.1.bias', 'module.conv5.3.bias', 'module.conv5.5.bias']:
        print('{:26} lr:  200 decay:0'.format(name)); net_parameters_id['conv5.bias'].append(param)
    elif name in ['module.sideOut1.weight', 'module.sideOut2.weight',
                  'module.sideOut3.weight', 'module.sideOut4.weight', 'module.sideOut5.weight']:
        print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id['score_dsn_1-5.weight'].append(param)
    elif name in ['module.sideOut1.bias', 'module.sideOut2.bias',
                  'module.sideOut3.bias', 'module.sideOut4.bias', 'module.sideOut5.bias']:
        print('{:26} lr: 0.02 decay:0'.format(name)); net_parameters_id['score_dsn_1-5.bias'].append(param)
    elif name in ['module.fuse.weight']:
        print('{:26} lr:0.001 decay:1'.format(name)); net_parameters_id['score_final.weight'].append(param)
    elif name in ['module.fuse.bias']:
        print('{:26} lr:0.002 decay:0'.format(name)); net_parameters_id['score_final.bias'].append(param)

# Create optimizer.
optimizer = torch.optim.SGD([
    {'params': net_parameters_id['conv1-4.weight']      , 'lr': learningRate*1    , 'weight_decay': weightDecay},
    {'params': net_parameters_id['conv1-4.bias']        , 'lr': learningRate*2    , 'weight_decay': 0.},
    {'params': net_parameters_id['conv5.weight']        , 'lr': learningRate*100  , 'weight_decay': weightDecay},
    {'params': net_parameters_id['conv5.bias']          , 'lr': learningRate*200  , 'weight_decay': 0.},
    {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': learningRate*0.01 , 'weight_decay': weightDecay},
    {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': learningRate*0.02 , 'weight_decay': 0.},
    {'params': net_parameters_id['score_final.weight']  , 'lr': learningRate*0.001, 'weight_decay': weightDecay},
    {'params': net_parameters_id['score_final.bias']    , 'lr': learningRate*0.002, 'weight_decay': 0.},
], lr=learningRate, momentum=momentum, weight_decay=weightDecay)

# Learning rate scheduler.
lr_schd = lr_scheduler.StepLR(optimizer, step_size=1e4, gamma=0.1)

print("Training started")

epochs = 15
i = 0
dispInterval = 500
lossAcc = 0.0
epoch_line = []
loss_line = []
optimizer.zero_grad()

for epoch in range(epochs):
    print("Epoch: " + str(epoch + 1))
    for j, data in enumerate(tqdm(train), 0):
        lr_schd.step()
        image, target = data
        image, target = Variable(image).cuda(), Variable(target).cuda()
        sideOuts = nnet(image)
        loss = sum([balanced_cross_entropy(sideOut, target) for sideOut in sideOuts[:-1]])
        loss6 = binary_cross_entropy(sideOuts[-1], target)
        loss += loss6
        lossAvg = loss/miniBatchSize
        lossAvg.backward()
        lossAcc += loss.item()
        optimizer.step()
        optimizer.zero_grad()    
        if (i+1) % dispInterval == 0:
            timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            lossDisp = lossAcc/dispInterval
            epoch_line.append(epoch + j/len(train))
            loss_line.append(lossDisp)
            print("%s epoch: %d iter:%d loss:%.6f"%(timestr, epoch+1, i+1, lossDisp))
            lossAcc = 0.0
        i += 1

    # transform to grayscale images
    avg = sum(sideOuts)/6
    side1 = grayTrans(sideOuts[0])
    side2 = grayTrans(sideOuts[1])
    side3 = grayTrans(sideOuts[2])
    side4 = grayTrans(sideOuts[3])
    side5 = grayTrans(sideOuts[4])
    fuse = grayTrans(sideOuts[5])
    avg = grayTrans(avg)
    tar = grayTrans(target)
        
    side1.save('images/sample_1_2.png')
    side2.save('images/sample_2_2.png')
    side3.save('images/sample_3_2.png')
    side4.save('images/sample_4_2.png')
    side5.save('images/sample_5_2.png')
    fuse.save('images/sample_6_2.png')
    avg.save('images/sample_7_2.png')
    tar.save('images/sample_T_2.png')

    torch.save(nnet.state_dict(), 'HED2.pth')

    plt.plot(epoch_line,loss_line)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("images/loss.png")


