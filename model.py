import torch 
from torch.nn.functional import upsample_bilinear as upsample
from torch.nn.functional import sigmoid

class HED(torch.nn.Module):
    def __init__(self):
        super(HED, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.sideOut1 = torch.nn.Conv2d(in_channels=64, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOut2 = torch.nn.Conv2d(in_channels=128, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOut3 = torch.nn.Conv2d(in_channels=256, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOut4 = torch.nn.Conv2d(in_channels=512, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOut5 = torch.nn.Conv2d(in_channels=512, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuse = torch.nn.Conv2d(in_channels=5, out_channels=1,
            kernel_size=1, stride=1, padding=0)

    def forward(self, image):

        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        height = image.size(2)
        width = image.size(3)

        sideOut1 = self.sideOut1(conv1)
        sideOut2 = upsample(self.sideOut2(conv2), size=(height,width))
        sideOut3 = upsample(self.sideOut3(conv3), size=(height,width))
        sideOut4 = upsample(self.sideOut4(conv4), size=(height,width))
        sideOut5 = upsample(self.sideOut5(conv5), size=(height,width))

        fuse = self.fuse(torch.cat((sideOut1, sideOut2, sideOut3, sideOut4, sideOut5), 1))

        sideOut1 = sigmoid(sideOut1)
        sideOut2 = sigmoid(sideOut2)
        sideOut3 = sigmoid(sideOut3)
        sideOut4 = sigmoid(sideOut4)
        sideOut5 = sigmoid(sideOut5)
        fuse = sigmoid(fuse)

        return sideOut1, sideOut2, sideOut3, sideOut4, sideOut5, fuse 