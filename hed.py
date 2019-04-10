import os
import sys
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
import time
from datetime import datetime
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from os.path import join, isdir, abspath, dirname

# Customized import.
from networks import HED
from datasets import BsdsDataset
from utils import Logger, AverageMeter, \
    load_checkpoint, save_checkpoint, load_vgg16_caffe, load_pretrained_caffe


# Parse arguments.
parser = argparse.ArgumentParser(description='HED training.')
# 1. Actions.
parser.add_argument('--test',             default=False,             help='Only test the model.', action='store_true')
# 2. Counts.
parser.add_argument('--train_batch_size', default=1,    type=int,   metavar='N', help='Training batch size.')
parser.add_argument('--test_batch_size',  default=1,    type=int,   metavar='N', help='Test batch size.')
parser.add_argument('--train_iter_size',  default=10,   type=int,   metavar='N', help='Training iteration size.')
parser.add_argument('--max_epoch',        default=40,   type=int,   metavar='N', help='Total epochs.')
parser.add_argument('--print_freq',       default=500,  type=int,   metavar='N', help='Print frequency.')
# 3. Optimizer settings.
parser.add_argument('--lr',               default=1e-6, type=float, metavar='F', help='Initial learning rate.')
parser.add_argument('--lr_stepsize',      default=1e4,  type=int,   metavar='N', help='Learning rate step size.')
# Note: Step size is based on number of iterations, not number of batches.
#   https://github.com/s9xie/hed/blob/94fb22f10cbfec8d84fbc0642b224022014b6bd6/src/caffe/solver.cpp#L498
parser.add_argument('--lr_gamma',         default=0.1,  type=float, metavar='F', help='Learning rate decay (gamma).')
parser.add_argument('--momentum',         default=0.9,  type=float, metavar='F', help='Momentum.')
parser.add_argument('--weight_decay',     default=2e-4, type=float, metavar='F', help='Weight decay.')
# 4. Files and folders.
parser.add_argument('--vgg16_caffe',      default='',                help='Resume VGG-16 Caffe parameters.')
parser.add_argument('--checkpoint',       default='',                help='Resume the checkpoint.')
parser.add_argument('--caffe_model',      default='',                help='Resume HED Caffe model.')
parser.add_argument('--output',           default='./output',        help='Output folder.')
parser.add_argument('--dataset',          default='./data/HED-BSDS', help='HED-BSDS dataset folder.')
# 5. Others.
parser.add_argument('--cpu',              default=False,             help='Enable CPU mode.', action='store_true')
args = parser.parse_args()

# Set device.
device = torch.device('cpu' if args.cpu else 'cuda')


def main():
    ################################################
    # I. Miscellaneous.
    ################################################
    # Create the output directory.
    current_dir = abspath(dirname(__file__))
    output_dir = join(current_dir, args.output)
    if not isdir(output_dir):
        os.makedirs(output_dir)

    # Set logger.
    now_str = datetime.now().strftime('%y%m%d-%H%M%S')
    log = Logger(join(output_dir, 'log-{}.txt'.format(now_str)))
    sys.stdout = log  # Overwrite the standard output.

    ################################################
    # II. Datasets.
    ################################################
    # Datasets and dataloaders.
    train_dataset = BsdsDataset(dataset_dir=args.dataset, split='train')
    test_dataset  = BsdsDataset(dataset_dir=args.dataset, split='test')
    train_loader  = DataLoader(train_dataset, batch_size=args.train_batch_size,
                               num_workers=4, drop_last=True, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=args.test_batch_size,
                               num_workers=4, drop_last=False, shuffle=False)

    ################################################
    # III. Network and optimizer.
    ################################################
    # Create the network in GPU.
    net = nn.DataParallel(HED(device))
    net.to(device)

    # Initialize the weights for HED model.
    def weights_init(m):
        """ Weight initialization function. """
        if isinstance(m, nn.Conv2d):
            # Initialize: m.weight.
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                # Constant initialization for fusion layer in HED network.
                torch.nn.init.constant_(m.weight, 0.2)
            else:
                # Zero initialization following official repository.
                # Reference: hed/docs/tutorial/layers.md
                m.weight.data.zero_()
            # Initialize: m.bias.
            if m.bias is not None:
                # Zero initialization.
                m.bias.data.zero_()
    net.apply(weights_init)

    # Optimizer settings.
    net_parameters_id = defaultdict(list)
    for name, param in net.named_parameters():
        if name in ['module.conv1_1.weight', 'module.conv1_2.weight',
                    'module.conv2_1.weight', 'module.conv2_2.weight',
                    'module.conv3_1.weight', 'module.conv3_2.weight', 'module.conv3_3.weight',
                    'module.conv4_1.weight', 'module.conv4_2.weight', 'module.conv4_3.weight']:
            print('{:26} lr:    1 decay:1'.format(name)); net_parameters_id['conv1-4.weight'].append(param)
        elif name in ['module.conv1_1.bias', 'module.conv1_2.bias',
                      'module.conv2_1.bias', 'module.conv2_2.bias',
                      'module.conv3_1.bias', 'module.conv3_2.bias', 'module.conv3_3.bias',
                      'module.conv4_1.bias', 'module.conv4_2.bias', 'module.conv4_3.bias']:
            print('{:26} lr:    2 decay:0'.format(name)); net_parameters_id['conv1-4.bias'].append(param)
        elif name in ['module.conv5_1.weight', 'module.conv5_2.weight', 'module.conv5_3.weight']:
            print('{:26} lr:  100 decay:1'.format(name)); net_parameters_id['conv5.weight'].append(param)
        elif name in ['module.conv5_1.bias', 'module.conv5_2.bias', 'module.conv5_3.bias'] :
            print('{:26} lr:  200 decay:0'.format(name)); net_parameters_id['conv5.bias'].append(param)
        elif name in ['module.score_dsn1.weight', 'module.score_dsn2.weight',
                      'module.score_dsn3.weight', 'module.score_dsn4.weight', 'module.score_dsn5.weight']:
            print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id['score_dsn_1-5.weight'].append(param)
        elif name in ['module.score_dsn1.bias', 'module.score_dsn2.bias',
                      'module.score_dsn3.bias', 'module.score_dsn4.bias', 'module.score_dsn5.bias']:
            print('{:26} lr: 0.02 decay:0'.format(name)); net_parameters_id['score_dsn_1-5.bias'].append(param)
        elif name in ['module.score_final.weight']:
            print('{:26} lr:0.001 decay:1'.format(name)); net_parameters_id['score_final.weight'].append(param)
        elif name in ['module.score_final.bias']:
            print('{:26} lr:0.002 decay:0'.format(name)); net_parameters_id['score_final.bias'].append(param)

    # Create optimizer.
    opt = torch.optim.SGD([
        {'params': net_parameters_id['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
        {'params': net_parameters_id['conv5.weight']        , 'lr': args.lr*100  , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id['conv5.bias']          , 'lr': args.lr*200  , 'weight_decay': 0.},
        {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
        {'params': net_parameters_id['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
        {'params': net_parameters_id['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Note: In train_val.prototxt and deploy.prototxt, the learning rates of score_final.weight/bias are different.

    # Learning rate scheduler.
    lr_schd = lr_scheduler.StepLR(opt, step_size=args.lr_stepsize, gamma=args.lr_gamma)

    ################################################
    # IV. Pre-trained parameters.
    ################################################
    # Load parameters from pre-trained VGG-16 Caffe model.
    if args.vgg16_caffe:
        load_vgg16_caffe(net, args.vgg16_caffe)

    # Resume the checkpoint.
    if args.checkpoint:
        load_checkpoint(net, opt, args.checkpoint)  # Omit the returned values.

    # Resume the HED Caffe model.
    if args.caffe_model:
        load_pretrained_caffe(net, args.caffe_model)

    ################################################
    # V. Training / testing.
    ################################################
    if args.test is True:
        # Only test.
        test(test_loader, net, save_dir=join(output_dir, 'test'))
    else:
        # Train.
        train_epoch_losses = []
        for epoch in range(args.max_epoch):
            # Initial test.
            if epoch == 0:
                print('Initial test...')
                test(test_loader, net, save_dir=join(output_dir, 'initial-test'))
            # Epoch training and test.
            train_epoch_loss = \
                train(train_loader, net, opt, lr_schd, epoch, save_dir=join(output_dir, 'epoch-{}-train'.format(epoch)))
            test(test_loader, net, save_dir=join(output_dir, 'epoch-{}-test'.format(epoch)))
            # Write log.
            log.flush()
            # Save checkpoint.
            save_checkpoint(state={'net': net.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch},
                            path=os.path.join(output_dir, 'epoch-{}-checkpoint.pt'.format(epoch)))
            # Collect losses.
            train_epoch_losses.append(train_epoch_loss)


def train(train_loader, net, opt, lr_schd, epoch, save_dir):
    """ Training procedure. """
    # Create the directory.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    # Switch to train mode and clear the gradient.
    net.train()
    opt.zero_grad()
    # Initialize meter and list.
    batch_loss_meter = AverageMeter()
    # Note: The counter is used here to record number of batches in current training iteration has been processed.
    #       It aims to have large training iteration number even if GPU memory is not enough. However, such trick
    #       can be used because batch normalization is not used in the network architecture.
    time_data = []
    time_network = []
    time_loss = []
    counter = 0
    for batch_index, (images, edges) in enumerate(tqdm(train_loader)):
        # Adjust learning rate and modify counter following Caffe's way.
        if counter == 0:
            lr_schd.step()  # Step at the beginning of the iteration.
        counter += 1
        if batch_index != 0:
            end = time.time()
            time_data.append(end - start)
        start = time.time()
        # Get images and edges from current batch.
        images, edges = images.to(device), edges.to(device)
        # Generate predictions.
        preds_list = net(images)
        end = time.time()
        time_network.append(end - start)
        # Calculate the loss of current batch (sum of all scales and fused).
        # Note: Here we mimic the "iteration" in official repository: iter_size batches will be considered together
        #       to perform one gradient update. To achieve the goal, we calculate the equivalent iteration loss
        #       eqv_iter_loss of current batch and generate the gradient. Then, instead of updating the weights,
        #       we continue to calculate eqv_iter_loss and add the newly generated gradient to current gradient.
        #       After iter_size batches, we will update the weights using the accumulated gradients and then zero
        #       the gradients.
        # Reference:
        #   https://github.com/s9xie/hed/blob/94fb22f10cbfec8d84fbc0642b224022014b6bd6/src/caffe/solver.cpp#L230
        #   https://www.zhihu.com/question/37270367
        start = time.time()
        batch_loss = sum([weighted_cross_entropy_loss(preds, edges) for preds in preds_list])
        eqv_iter_loss = batch_loss / args.train_iter_size

        # Generate the gradient and accumulate (using equivalent average loss).
        eqv_iter_loss.backward()
        end = time.time()
        time_loss.append(end - start)
        if counter == args.train_iter_size:
            opt.step()
            opt.zero_grad()
            counter = 0  # Reset the counter.
            print("Loss time: ", np.average(time_loss))
            print("Network time: ", np.average(time_network))
            print("Data time: ", np.average(time_data))
        # Record loss.
        batch_loss_meter.update(batch_loss.item())
        # Log and save intermediate images.
        if batch_index % args.print_freq == args.print_freq - 1:
            # Log.
            print(('Training epoch:{}/{}, batch:{}/{} current iteration:{}, ' +
                   'current batch batch_loss:{}, epoch average batch_loss:{}, learning rate list:{}.').format(
                   epoch, args.max_epoch, batch_index, len(train_loader), lr_schd.last_epoch,
                   batch_loss_meter.val, batch_loss_meter.avg, lr_schd.get_lr()))
            # Generate intermediate images.
            preds_list_and_edges = preds_list + [edges]
            _, _, h, w = preds_list_and_edges[0].shape
            interm_images = torch.zeros((len(preds_list_and_edges), 1, h, w))
            for i in range(len(preds_list_and_edges)):
                # Only fetch the first image in the batch.
                interm_images[i, 0, :, :] = preds_list_and_edges[i][0, 0, :, :]
            # Save the images.
            torchvision.utils.save_image(interm_images, join(save_dir, 'batch-{}-1st-image.png'.format(batch_index)))
    # Return the epoch average batch_loss.
    return batch_loss_meter.avg


def test(test_loader, net, save_dir):
    """ Test procedure. """
    # Create the directories.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    save_png_dir = join(save_dir, 'png')
    if not isdir(save_png_dir):
        os.makedirs(save_png_dir)
    save_mat_dir = join(save_dir, 'mat')
    if not isdir(save_mat_dir):
        os.makedirs(save_mat_dir)
    # Switch to evaluation mode.
    net.eval()
    # Generate predictions and save.
    assert args.test_batch_size == 1  # Currently only support test batch size 1.
    for batch_index, images in enumerate(tqdm(test_loader)):
        images = images.cuda()
        _, _, h, w = images.shape
        preds_list = net(images)
        fuse       = preds_list[-1].detach().cpu().numpy()[0, 0]  # Shape: [h, w].
        name       = test_loader.dataset.images_name[batch_index]
        sio.savemat(join(save_mat_dir, '{}.mat'.format(name)), {'result': fuse})
        Image.fromarray((fuse * 255).astype(np.uint8)).save(join(save_png_dir, '{}.png'.format(name)))
        # print('Test batch {}/{}.'.format(batch_index + 1, len(test_loader)))


def weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
    num_neg = c * h * w - num_pos                     # Shape: [b,].
    weight = torch.zeros_like(mask)
    weight[edges > 0.5]  = num_neg / (num_pos + num_neg)
    weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    # Calculate loss.
    losses = torch.nn.functional.binary_cross_entropy(preds.float(), edges.float(), weight=weight, reduction='none')
    loss   = torch.sum(losses) / b
    return loss


if __name__ == '__main__':
    main()
