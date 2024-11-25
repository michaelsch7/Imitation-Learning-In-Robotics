import torch
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch import flatten
import torch.nn as nn
from torch.nn.functional import relu

class RoboNet(Module):
    
    def __init__(self, classes) -> None:
        super(RoboNet, self).__init__()
        
        self.convLayer1 = Conv2d(in_channels=3, out_channels=32, kernel_size=(4,4))
        self.maxpoolLayer1 = MaxPool2d(kernel_size=(3,3), stride=(4,4))
        self.convLayer2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4))
        self.maxpoolLayer2 = MaxPool2d(kernel_size=(3,3), stride=(4,4))
        
        self.fc1 = Linear(in_features= 53824 + 4, out_features=64) # +4 is for Arm Input
        self.fc2 = Linear(in_features=64, out_features=32)
        self.fc3 = Linear(in_features=32, out_features=classes)
    
  
    def forward(self, inp, pos_grip):
        
        x = self.maxpoolLayer1(relu(self.convLayer1(inp)))
        x = self.maxpoolLayer2(relu(self.convLayer2(x)))
      
        x = flatten(x, 1)
        x = torch.cat((x, pos_grip.float()*35), dim=1)
        
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
