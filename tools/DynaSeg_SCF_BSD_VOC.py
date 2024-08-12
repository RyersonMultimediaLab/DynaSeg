import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
import torch.nn.init
import os
from os.path import join, splitext

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Dynamically Weighted Loss for Unsupervised Image Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--input_folder', metavar='FOLDER',
                    help='input image folder path', required=True)
parser.add_argument('--output_folder', metavar='FOLDER',
                    help='output image folder path', required=True)
args = parser.parse_args()

# Set up device
device = torch.device("cuda" if use_cuda else "cpu")

# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv - 1):
            self.conv2.append(nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(args.nChannel))
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(args.nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# Loop through all files in the input folder
input_folder = args.input_folder
output_folder = args.output_folder

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Add more image extensions if needed
        input_path = join(input_folder, filename)
        output_path = join(output_folder, splitext(filename)[0] + '.png')

        # Load image
        im = cv2.imread(input_path)
        data = torch.from_numpy(np.array([im.transpose(2, 0, 1).astype('float32') / 255.]))
        if use_cuda:
            data = data.cuda()
        data = Variable(data)

        # Initialize model
        model = MyNet(data.size(1)).to(device)

        # similarity loss definition
        loss_fn = torch.nn.CrossEntropyLoss()

        # Define optimizer
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        label_colours = np.random.randint(255, size=(100, 3))

        # Train the model
        for batch_idx in range(args.maxIter):
            # forwarding
            optimizer.zero_grad()
            output = model(data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

            outputHP = output.reshape((im.shape[0], im.shape[1], args.nChannel))
            lhpy = F.l1_loss(outputHP[1:, :, :], outputHP[0:-1, :, :])
            lhpz = F.l1_loss(outputHP[:, 1:, :], outputHP[:, 0:-1, :])

            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))
            if args.visualize:
                im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
                im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
                cv2.imshow("output", im_target_rgb)
                cv2.waitKey(10)

            # SCF loss
            loss = loss_fn(output, target) + nLabels/15 * (lhpy + lhpz)

            # FSF loss
            # loss = loss_fn(output, target) + 50/nLabels * (lhpy + lhpz)

            loss.backward()
            optimizer.step()

            print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

            if nLabels <= args.minLabels:
                print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
                break

        # Save segmented image
        if not args.visualize:
            output = model(data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        cv2.imwrite(output_path, im_target_rgb)
        print(f"Saved segmented image: {output_path}")