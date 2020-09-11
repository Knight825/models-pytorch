import torch
import torchvision
from torchvision.models.resnet import conv1x1,conv3x3,BasicBlock,Bottleneck
from expendoperation import _Add_DifferentChannels

class PrnBasicBlock(BasicBlock):
    expansion = 1
    def __init__(self,inplanes,planes,residualrate = 0.5,stride=1,downsample = None,groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(PrnBasicBlock,self).__init__(inplanes,planes,stride,downsample,groups,
                                           base_width, dilation, norm_layer)
        self.residualchannel = int(inplanes*residualrate)
        if self.residualchannel > planes:
            raise ValueError('partial residual channel set worry')
    def forward(self,x):
        identity = x[:,:self.residualchannel,...]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  

        if self.downsample is not None:
            identity = self.downsample(identity)  

        out = _Add_DifferentChannels(identity,out)
        out = self.ReLU(out)

        return out      


class PrnBottleneck(Bottleneck):
    expansion = 4
    def __init__(self,inplanes,planes,residualrate = 0.5,stride=1,downsample = None,groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(PrnBottleneck,self).__init__(inplanes,planes,stride,downsample,groups,
                                           base_width, dilation, norm_layer)
        self.residualchannel = int(inplanes*residualrate)
        if self.residualchannel > planes * self.expansion:
            raise ValueError('partial residual channel set worry')

    def forward(self,x):
        identity = x[:,:self.residualchannel,...]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = _Add_DifferentChannels(identity,out)     
        out = self.relu(out)

        return out           


class Prn_ResNet(torch.nn.Module):
    def __init__(self, block, layers,partialresidualrates = [0.5,0.5,0.5,0.5] ,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Prn_ResNet,self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], partialresidualrates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], partialresidualrates[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], partialresidualrates[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], partialresidualrates[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PrnBottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, PrnBasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)
    def _make_layer(self,block, planes, blocks, partialresidualrate, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, self.inplanes, stride),
                norm_layer(self.inplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, 1.0, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, partialresidualrate, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return torch.nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def _prn_resnet(block,layers,partialresidualrates,**kwargs):
    model = Prn_ResNet(block,layers,partialresidualrates,**kwargs)
    return model

def prn_resnet18(partialresidualrates = [0.5,0.5,0.5,0.5]):
    return _prn_resnet(PrnBasicBlock,[2,2,2,2],partialresidualrates)

def prn_resnet34(partialresidualrates = [0.5,0.5,0.5,0.5]):
    return _prn_resnet(PrnBasicBlock,[3, 4, 6, 3],partialresidualrates)

def prn_resnet50(partialresidualrates = [0.5,0.5,0.5,0.5]):
    return _prn_resnet(PrnBottleneck,[3, 4, 6, 3],partialresidualrates)

def prn_resnet101(partialresidualrates = [0.5,0.5,0.5,0.5]):
    return _prn_resnet(PrnBottleneck,[3, 4, 23, 3],partialresidualrates)

def prn_resnet152(partialresidualrates = [0.5,0.5,0.5,0.5]):
    return _prn_resnet(PrnBottleneck,[3, 8, 36, 3],partialresidualrates)