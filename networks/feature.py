import torch 
import torch.nn as nn 
from networks.helper import conv2Relu
from torchvision.models import vgg19

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        '''通常の畳み込み層'''
        self.vgg = vgg19(pretrained=True).features[0:23] # -> (b, 512, 46, 46)
        self.layer1 = conv2Relu(512, 256, 3, 1, 1)
        self.layer2 = conv2Relu(256, 128, 3, 1, 1)

    def forward(self, x):
        '''
        (b, 3, 368, 368) -> (b, 128, 46, 46)
        '''
        y = self.vgg(x)
        y = self.layer1(y)
        y = self.layer2(y)
        return y 