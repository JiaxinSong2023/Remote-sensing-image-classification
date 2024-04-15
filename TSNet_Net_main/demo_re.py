# from __future__ import print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt

from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from torch.utils.data import DataLoader

from models.MyNet import MyNet
from models.TSNet2 import TSNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


scribble = False
nChannel = 100
maxIter = 1000
minLabels = 8
lr = 0.001                  #############################################################################################
nConv = 2
visualize = 1
stepsize_sim = 1
stepsize_con = 0.5
stepsize_scr = 0.5

# from ST import SwinTransformer
# model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=12,
#                             embed_dim=192,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(6, 12, 24, 48),
#                             num_classes=nChannel).to(device=DEVICE)
#
# model.load_state_dict(torch.load('2023-02-25-14-17-15-storage-swin_large_patch4_window12_384_22k.pth'), strict=False)

# model = MyNet(3).to(device=DEVICE)

# from models.u_ASPP_net import UNet
# model = UNet(num_classes=nChannel).to(device=DEVICE)


from models.unet import UNet
# model = UNet(num_classes=nChannel).to(device=DEVICE)
model = TSNet(n_class=100).cuda()

im = np.array(Image.open(r'E:\remote-sensing image\shangtang\train\im2\04997.png'))
# im = im1

Ma, Na, _ = im.shape

data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))

# data = torch.cat([data, data], dim=0)
data = data.to(device=DEVICE)

# ima = match_histograms(ima, imb, channel_axis=True)     # 被匹配图像， mask

# im = cv2.resize(im, (384, 384), interpolation=cv2.INTER_CUBIC)

# load scribble
if scribble:
    mask = cv2.imread(input.replace('.' + input.split('.')[-1], '_scribble.png'), -1)
    mask = mask.reshape(-1)
    mask_inds = np.unique(mask)
    mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
    inds_sim = torch.from_numpy(np.where(mask == 255)[0])
    inds_scr = torch.from_numpy(np.where(mask != 255)[0])
    target_scr = torch.from_numpy(mask.astype(np.int))

    inds_sim = inds_sim.to(device=DEVICE)
    inds_scr = inds_scr.to(device=DEVICE)
    target_scr = target_scr.to(device=DEVICE)
    target_scr = Variable(target_scr)
    # set minLabels
    minLabels = len(mask_inds)

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# scribble loss definition
loss_fn_scr = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average=True)
loss_hpz = torch.nn.L1Loss(size_average=True)

HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], nChannel)
HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, nChannel)

HPy_target = HPy_target.to(device=DEVICE)
HPz_target = HPz_target.to(device=DEVICE)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))

start = time.time()

for batch_idx in range(maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)

    outputHP = output.reshape((im.shape[0], im.shape[1], nChannel))
    HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
    HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
    lhpy = loss_hpy(HPy, HPy_target)
    lhpz = loss_hpz(HPz, HPz_target)

    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))
    if visualize:
        im_target_rgb = np.array([label_colours[c % nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        # im_target_rgb_con = np.vstack((im, im_target_rgb))
        # cv2.imwrite('E:/deepcluster/label_rgb/' + str(minLabels) + imagename, im_target_rgb_con)
        cv2.imshow("output", im_target_rgb)
        # cv2.waitKey(10)


        im_target_gray = im_target.reshape(im.shape[:2])

        # if np.log2(batch_idx) % 1 == 0:
        #     cv2.imwrite("output_%s_%02i.jpg" % (args.input, batch_idx), im_target_rgb)

        label_figures = np.unique(im_target)
        label_image = np.zeros(im.shape[:2])

        dict = {}
        for num, li in enumerate(label_figures):
            dict[li] = num

        for idxi in range(im.shape[0]):
            for idxj in range(im.shape[1]):
                label_image[idxi, idxj] = dict[im_target_gray[idxi, idxj]]

        # label_image_a = label_image[: Ma, : int(Na / 2)]
        # label_image_b = label_image[: Ma, int(Na / 2):]
        #
        # cv2.imwrite('E:/deepcluster/label/im1/' + str(minLabels) + imagename, label_image_a)
        # cv2.imwrite('E:/deepcluster/label/im2/' + str(minLabels) + imagename, label_image_b)


    # loss
    if scribble:
        loss = stepsize_sim * loss_fn(output[inds_sim], target[inds_sim]) + stepsize_scr * loss_fn_scr(
            output[inds_scr], target_scr[inds_scr]) + stepsize_con * (lhpy + lhpz)
    else:
        loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)
        # loss = loss_fn(output, target)

    loss.backward()
    optimizer.step()

    print(batch_idx, '/', maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

    if nLabels <= minLabels:
        print("nLabels", nLabels, "reached minLabels", minLabels, ".")
        break

endtime = time.time()
print(endtime - start)
# save output image
if not visualize:
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
# cv2.imwrite("output.png", im_target_rgb)

plt.imshow(label_image)
plt.show()
