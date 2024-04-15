
# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from PIL import Image

from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from torch.utils.data import DataLoader

from sklearn import metrics

from models.MyNet import MyNet
from models.TSNet2 import TSNet
from models.unet import UNet
from sklearn import metrics


def UNetUnsupervised(im, minLabels=4, lr=0.1, maxIter=100, u=0.5):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cpu')
    nChannel = 100
    visualize = 1
    stepsize_sim = 1
    stepsize_con = u

    # model = MyNet(3).to(device=DEVICE)
    # model = UNet(num_classes=nChannel).to(device=DEVICE)
    model = TSNet(n_class=100).cuda()

    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
    data = data.to(device=DEVICE)

    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(size_average=True)
    loss_hpz = torch.nn.L1Loss(size_average=True)

    HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], nChannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, nChannel)

    HPy_target = HPy_target.to(device=DEVICE)
    HPz_target = HPz_target.to(device=DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    label_colours = np.random.randint(255, size=(100, 3))

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
            im_target_rgb_con = np.hstack((im, im_target_rgb))
            im_target_gray = im_target.reshape(im.shape[:2])

            cv2.imshow("output", im_target_rgb)
            cv2.waitKey(10)

            label_figures = np.unique(im_target)
            label_image = np.zeros(im.shape[:2], dtype=np.uint8)

            dict = {}
            for num, li in enumerate(label_figures):
                dict[li] = num

            for idxi in range(im.shape[0]):
                for idxj in range(im.shape[1]):
                    label_image[idxi, idxj] = dict[im_target_gray[idxi, idxj]]

        # loss
        loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)
        loss.backward()
        optimizer.step()

        # print(batch_idx, '/', maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())
        print(loss.item())

        if nLabels <= minLabels:
            print("nLabels", nLabels, "reached minLabels", minLabels, ".")
            break

    return label_image, im_target_rgb, im_target_rgb_con


if __name__ == '__main__':

    # 读取真实标签
    name = '1'
    u = 1

    im = np.array(Image.open(r'E:\remote-sensing image\2021LoveDA\Train\images_png\1.png'))
    label_image, im_target_rgb, im_target_rgb_con = UNetUnsupervised(im, minLabels=6, lr=0.0001, u=u)

    plt.imshow(label_image)
    plt.show()










