import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.stem_conv = nn.Conv2d(119, 256, kernel_size=(3,3), padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256, 256) for _ in range(19)])
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn(self.stem_conv(x)))
        x = self.residual_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))

        return F.relu(x + h)
        
class PolicyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 2, kernel_size=(1,1), bias=False)
        self.conv2 = nn.Conv2d(2, 73, kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv1(x)))
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1) # (batch_size, 2, 8, 8) -> (batch_size, 128)
        return x
    

class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=(1,1), bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = torch.tanh(x)

        return x
    