"""Binance model."""
import torch
import torch.nn as nn


def init_module(layer):
    """Initial modules weights and biases."""
    for m in layer.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            m.weight.data.uniform_()
            if m.bias is not None:
                m.bias.data.zero_()


class Buy_Model(nn.Module):
    """Buy model for Binance."""

    def __init__(self, input_dim):
        """Init."""
        super(Buy_Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer2 = nn.Linear(100, 30)
        self.layer3 = nn.Linear(30, 2)
        self.drop = nn.Dropout(0.2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        """Init model weights."""
        init_module(self.layer_1)
        init_module(self.layer_2)
        init_module(self.layer_3)

    def forward(self, x):
        """Forward."""
        x = self.relu(self.layer1(x))
        x = self.drop(x)
        x = self.relu(self.layer2(x))
        x = self.drop(x)
        x = self.softmax(self.layer3(x))
        return x
