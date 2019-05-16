import torch 
from torch.nn.functional import interpolate
from torch import sigmoid
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if m.in_channels == 5:
            torch.nn.init.constant_(m.weight.data,0.2)
        else:
            torch.nn.init.constant_(m.weight.data,0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data,0)

def initialize_hed(path, continue_train, path_HED=None):
    net = HED()
    if continue_train:
        dic = torch.load(path_HED)
        dicli = list(dic.keys())
        new = {}
        j = 0
        for k in net.state_dict():
            new[k] = dic[dicli[j]]
            j += 1
        net.load_state_dict(new)
        return net
    else:
        vgg16_items = list(torch.load(path).items())
        net.apply(weights_init)
        j = 0
        for k, v in net.state_dict().items():
            if k.find("conv") != -1:
                net.state_dict()[k].copy_(vgg16_items[j][1])
                j += 1
        return net

class HED(torch.nn.Module):
    def __init__(self):
        super(HED, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                stride=1, padding=35),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
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
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
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
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
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

        # Fixed bilinear weights.
        self.weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        self.weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        self.weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        self.weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        # Prepare for aligned crop.
        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = \
            self.prepare_aligned_crop()

    # noinspection PyMethodMayBeStatic
    def prepare_aligned_crop(self):
        """ Prepare for aligned crop. """
        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

        def map_inv(m):
            """ Mapping inverse. """
            a, b = m
            return 1 / a, -b / a

        def map_compose(m1, m2):
            """ Mapping compose. """
            a1, b1 = m1
            a2, b2 = m2
            return a1 * a2, a1 * b2 + b1

        def deconv_map(kernel_h, stride_h, pad_h):
            """ Deconvolution coordinates mapping. """
            return stride_h, (kernel_h - 1) / 2 - pad_h

        def conv_map(kernel_h, stride_h, pad_h):
            """ Convolution coordinates mapping. """
            return map_inv(deconv_map(kernel_h, stride_h, pad_h))

        def pool_map(kernel_h, stride_h, pad_h):
            """ Pooling coordinates mapping. """
            return conv_map(kernel_h, stride_h, pad_h)

        x_map = (1, 0)
        conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
        conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
        pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

        conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
        conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
        pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

        conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
        conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
        conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
        pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

        conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
        conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
        conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
        pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

        conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
        conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
        conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

        score_dsn1_map = conv1_2_map
        score_dsn2_map = conv2_2_map
        score_dsn3_map = conv3_3_map
        score_dsn4_map = conv4_3_map
        score_dsn5_map = conv5_3_map

        upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
        upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
        upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
        upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

        crop1_margin = int(score_dsn1_map[1])
        crop2_margin = int(upsample2_map[1])
        crop3_margin = int(upsample3_map[1])
        crop4_margin = int(upsample4_map[1])
        crop5_margin = int(upsample5_map[1])

        return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin

    def forward(self, image):

        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        height = image.size(2)
        width = image.size(3)

        score_dsn1 = self.sideOut1(conv1)
        score_dsn2 = self.sideOut2(conv2)
        score_dsn3 = self.sideOut3(conv3)
        score_dsn4 = self.sideOut4(conv4)
        score_dsn5 = self.sideOut5(conv5)

        upsample2 = torch.nn.functional.conv_transpose2d(score_dsn2, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(score_dsn3, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(score_dsn4, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(score_dsn5, self.weight_deconv5, stride=16)

        # Aligned cropping.
        sideOut1 = score_dsn1[:, :, self.crop1_margin:self.crop1_margin+height,
                                 self.crop1_margin:self.crop1_margin+width]
        sideOut2 = upsample2[:, :, self.crop2_margin:self.crop2_margin+height,
                                self.crop2_margin:self.crop2_margin+width]
        sideOut3 = upsample3[:, :, self.crop3_margin:self.crop3_margin+height,
                                self.crop3_margin:self.crop3_margin+width]
        sideOut4 = upsample4[:, :, self.crop4_margin:self.crop4_margin+height,
                                self.crop4_margin:self.crop4_margin+width]
        sideOut5 = upsample5[:, :, self.crop5_margin:self.crop5_margin+height,
                                self.crop5_margin:self.crop5_margin+width]

        fuse = self.fuse(torch.cat((sideOut1, sideOut2, sideOut3, sideOut4, sideOut5), 1))

        sigSideOut1 = sigmoid(sideOut1)
        sigSideOut2 = sigmoid(sideOut2)
        sigSideOut3 = sigmoid(sideOut3)
        sigSideOut4 = sigmoid(sideOut4)
        sigSideOut5 = sigmoid(sideOut5)
        fuse = sigmoid(fuse)

        return sigSideOut1, sigSideOut2, sigSideOut3, sigSideOut4, sigSideOut5, fuse 

def make_bilinear_weights(size, num_channels):
    """ Generate bi-linear interpolation weights as up-sampling filters (following FCN paper). """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False  # Set not trainable.
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w
