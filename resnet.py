import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        ############################
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        ############################
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        # self.pool3 = nn.AdaptiveAvgPool2d((None, 2))
        # self.pool4 = nn.AdaptiveAvgPool2d((2, None))
        # self.pool5 = nn.AdaptiveAvgPool2d((None, 3))
        # self.pool6 = nn.AdaptiveAvgPool2d((3, None))
#         self.pool7 = nn.AdaptiveAvgPool2d((None, 4))
#         self.pool8 = nn.AdaptiveAvgPool2d((4, None))
        ############################
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        #print(x.shape)
        _, _, h, w = x.size()
        self.pool3 = nn.AdaptiveAvgPool2d((1, w-1))
        self.pool4 = nn.AdaptiveAvgPool2d((h-1, 1))
        self.pool5 = nn.AdaptiveAvgPool2d((1, w-2))
        self.pool6 = nn.AdaptiveAvgPool2d((h-2, 1))
        x1 = self.pool1(x)
        #print(x1.shape)
        x1 = self.conv1(x1)
        #print(x1.shape)
        x1 = self.bn1(x1)
        #print(x1.shape)
        x1 = x1.expand(-1, -1, h, w)
        #print(x1.shape)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x3 = self.pool3(x)
        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = x3.expand(-1, -1, h, w-1)
        x3 = F.interpolate(x3, (h, w))

        x4 = self.pool4(x)
        x4 = self.conv2(x4)
        x4 = self.bn2(x4)
        x4 = x4.expand(-1, -1, h-1, w)
        x4 = F.interpolate(x4, (h, w))

        x5 = self.pool5(x)
        x5 = self.conv1(x5)
        x5 = self.bn1(x5)
        x5 = x5.expand(-1, -1, h, w-2)
        x5 = F.interpolate(x5, (h, w))

        x6 = self.pool6(x)
        x6 = self.conv2(x6)
        x6 = self.bn2(x6)
        x6 = x6.expand(-1, -1, h-2, w)
        x6 = F.interpolate(x6, (h, w))
        
#         x7 = self.pool7(x)
#         x7 = self.conv1(x7)
#         x7 = self.bn1(x7)
#         #x5 = x5.expand(-1, -1, h, w)
#         x7 = F.interpolate(x7, (h, w))

#         x8 = self.pool8(x)
#         x8 = self.conv2(x8)
#         x8 = self.bn2(x8)
#         #x6 = x6.expand(-1, -1, h, w)
#         x8 = F.interpolate(x8, (h, w))
        
        x = self.relu(x1 + x2 + x3 + x4 + x5 + x6)
        x = self.conv3(x).sigmoid()
        return x



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=None, spm_on=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.spm = None
        if spm_on:
            self.spm = SPBlock(planes, planes, norm_layer=norm_layer)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.spm is not None:
            out = out * self.spm(out) #add SPM after the first Conv3x3

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        
        spm_on = False
        if planes == 512:
            spm_on = True

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer, spm_on=spm_on))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=4, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer, spm_on=spm_on))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i >= blocks - 1 or planes == 512:
                spm_on = True
            layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation, norm_layer=norm_layer, spm_on=spm_on))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    model.load_state_dict(torch.load('weights/resnet34_pytorch.pth'), strict=False)
    return model
 