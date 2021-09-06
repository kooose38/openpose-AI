import torch 
import torch.nn as nn 

from networks.feature import Feature 
from networks.helper import conv2Relu 


class Stage1(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(Stage1, self).__init__()
        self.layer1 = conv2Relu(in_c, in_c, 3, 1, 1)
        self.layer2 = conv2Relu(in_c, in_c, 3, 1, 1)
        self.layer3 = conv2Relu(in_c, in_c, 3, 1, 1)
        self.layer4 = conv2Relu(in_c, mid_c, 1, 1, 0)
        self.layer5 = nn.Conv2d(mid_c, out_c, 1, 1, padding=0)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        return y 

class Stage2(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(Stage2, self).__init__()
        self.layer1 = conv2Relu(in_c, mid_c, 7, 1, 3)
        self.layer2 = conv2Relu(mid_c, mid_c, 7, 1, 3)
        self.layer3 = conv2Relu(mid_c, mid_c, 7, 1, 3)
        self.layer4 = conv2Relu(mid_c, mid_c, 7, 1, 3)
        self.layer5 = conv2Relu(mid_c, mid_c, 7, 1, 3)
        self.layer6 = conv2Relu(mid_c, mid_c, 1, 1, 0)
        self.layer7 = nn.Conv2d(mid_c, out_c, 1, 1, padding=0)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        y = self.layer6(y)
        y = self.layer7(y)
        return y 

class OpenPoseNet(nn.Module):
    def __init__(self):
        super(OpenPoseNet, self).__init__()
        concat_size = 128+38+19
        self.feature = Feature()
        # pafs層 (b, 38, 46, 46)
        self.module1_1 = Stage1(128, 512, 38)
        self.module1_2 = Stage2(concat_size, 128, 38)
        self.module1_3 = Stage2(concat_size, 128, 38)
        self.module1_4 = Stage2(concat_size, 128, 38)
        self.module1_5 = Stage2(concat_size, 128, 38)
        self.module1_6 = Stage2(concat_size, 128, 38)
        # heatmap層 (b, 19, 6, 46)
        self.module2_1 = Stage1(128, 512, 19)
        self.module2_2 = Stage2(concat_size, 128, 19)
        self.module2_3 = Stage2(concat_size, 128, 19)
        self.module2_4 = Stage2(concat_size, 128, 19)
        self.module2_5 = Stage2(concat_size, 128, 19)
        self.module2_6 = Stage2(concat_size, 128, 19)

        self.outputs = []

    def forward(self, x):
        out = self.feature(x)

        out1_1 = self.module1_1(out)
        out1_2 = self.module2_1(out)
        out1 = torch.cat((out, out1_1, out1_2), dim=1)

        out2_1 = self.module1_2(out1)
        out2_2 = self.module2_2(out1)
        out2 = torch.cat((out, out2_1, out2_2), dim=1)

        out3_1 = self.module1_3(out2)
        out3_2 = self.module2_3(out2)
        out3 = torch.cat((out, out3_1, out3_2), dim=1)

        out4_1 = self.module1_4(out3)
        out4_2 = self.module2_4(out3)
        out4 = torch.cat((out, out4_1, out4_2), dim=1)

        out5_1 = self.module1_5(out4)
        out5_2 = self.module2_5(out4)
        out5 = torch.cat((out, out5_1, out5_2), dim=1)

        out6_1 = self.module1_6(out5)
        out6_2 = self.module2_6(out5)

        self.outputs.append(out1_1)
        self.outputs.append(out1_2)
        self.outputs.append(out2_1)
        self.outputs.append(out2_2)
        self.outputs.append(out3_1)
        self.outputs.append(out3_2)
        self.outputs.append(out4_1)
        self.outputs.append(out4_2)
        self.outputs.append(out5_1)
        self.outputs.append(out5_2)
        self.outputs.append(out6_1)
        self.outputs.append(out6_2)

        return self.outputs 