import torch 
import torch.nn as nn 

class conv2Relu(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding):
        super(conv2Relu, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))