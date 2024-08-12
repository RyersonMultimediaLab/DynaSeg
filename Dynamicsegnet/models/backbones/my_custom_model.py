# my_custom_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyCustomModel(nn.Module):
    def __init__(self, M, p, qdy):
        super(MyCustomModel, self).__init__()
        self.M = M
        self.p = p
        self.qdy = qdy

        # Define the convolutional components
        self.conv_components = nn.ModuleList()
        for _ in range(M):
            conv = nn.Conv2d(p, p, kernel_size=3, stride=1, padding=1)
            relu = nn.ReLU(inplace=True)
            bn = nn.BatchNorm2d(p)
            self.conv_components.append(nn.Sequential(conv, relu, bn))

        # Final classification layer
        self.classifier = nn.Conv2d(p, qdy, kernel_size=1, stride=1)

        # Batch normalization
        self.bn = nn.BatchNorm2d(p)

    def forward(self, x):
        # CNN subnetwork
        r = x
        for conv_component in self.conv_components:
            r = conv_component(r)

        # Normalize the response map
        rdy = self.bn(r)

        # Classification
        logits = self.classifier(rdy)

        # Argmax to obtain cluster labels
        _, labels = torch.max(rdy, dim=1)

        return logits, labels


class MyCustomLoss(nn.Module):
    def __init__(self, qdy, mu):
        super(MyCustomLoss, self).__init__()
        self.qdy = qdy
        self.mu = mu

    def forward(self, rdy, labels):
        # Feature similarity loss
        loss_sim = F.cross_entropy(rdy, labels)

        # Spatial continuity loss
        diff_h = torch.abs(rdy[:, :, 1:, :] - rdy[:, :, :-1, :])
        diff_v = torch.abs(rdy[:, :, :, 1:] - rdy[:, :, :, :-1])
        loss_con = torch.mean(diff_h) + torch.mean(diff_v)

        # Total loss
        loss = loss_sim + self.mu * loss_con / self.qdy

        return loss
