import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Network(nn.Module):

    def compute_fc_size(self, num_channels, height, width):
        # Применение сверточных и пулинг слоев для вычисления размера входа
        x = torch.rand(1, num_channels, height, width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = F.dropout(x, 0.1)

        # Вычисление размера входа для полносвязанного слоя
        fc_size = x.view(1, -1).size(1)
        # print("Size of layer after convolution layers", fc_size)
        return fc_size

    def __init__(self, n_actions):

        super().__init__()
        
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)

        
        self.fc_size = self.compute_fc_size(4, 64, 64)
        print("fc_size", self.fc_size)

        self.fc1 = nn.Linear(self.fc_size, 512)
        self.fc2 = nn.Linear(512, n_actions)


    def forward(self, state_t):
        state_t = torch.as_tensor(state_t)
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        x = self.conv1(state_t)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, self.fc_size)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
