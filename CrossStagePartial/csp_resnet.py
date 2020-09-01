import torch
from torchvision.models.resnet import conv1x1,conv3x3,BasicBlock,Bottleneck,ResNet


def _downsample(inplanes,outplanes,stride):
    return torch.nn.Sequential(
        conv1x1(inplanes, outplanes, stride),
        torch.nn.BatchNorm2d(outplanes),
    )


class Csp_ResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,channldown = True):
        self.expansion = block.expansion
        self.layerplanes = [64]
        self.CspChanneldown = 2 if channldown else 1
        super(Csp_ResNet,self).__init__(block, layers, num_classes, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.downlayer1 = torch.nn.Identity()
        self.downlayer2 = _downsample(self.layerplanes[1]//2,self.layerplanes[1]//2,2)
        self.downlayer3 = _downsample(self.layerplanes[2]//2,self.layerplanes[2]//2,2)
        self.downlayer4 = _downsample(self.layerplanes[3]//2,self.layerplanes[3]//2,2)
        self.fc = torch.nn.Linear(self.layerplanes[4],num_classes)



    def _make_layer(self,block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        inplanes = self.inplanes - self.inplanes//2
        outplanes = (planes //self.CspChanneldown) * block.expansion
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != outplanes:
            downsample = torch.nn.Sequential(
                conv1x1(inplanes, outplanes, stride),
                norm_layer(outplanes),
            )
        layers = []
        layers.append(block(inplanes, planes//self.CspChanneldown, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        inplanes = outplanes
        for _ in range(1,blocks):
            layers.append(block(inplanes, planes//self.CspChanneldown, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        self.inplanes = self.inplanes//2 +  outplanes
        self.layerplanes.append(self.inplanes)
        return torch.nn.Sequential(*layers)           
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.cat([self.downlayer1(x[:,:self.layerplanes[0]//2,...]),self.layer1(x[:,self.layerplanes[0]//2:,...])],1)   
        x = torch.cat([self.downlayer2(x[:,:self.layerplanes[1]//2,...]),self.layer2(x[:,self.layerplanes[1]//2:,...])],1)   
        x = torch.cat([self.downlayer3(x[:,:self.layerplanes[2]//2,...]),self.layer3(x[:,self.layerplanes[2]//2:,...])],1)
        x = torch.cat([self.downlayer4(x[:,:self.layerplanes[3]//2,...]),self.layer4(x[:,self.layerplanes[3]//2:,...])],1) 

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)   
        return x  


def _csp_resnet(block, layers, **kwargs):
    model = Csp_ResNet(block,layers,**kwargs)
    return model


def csp_resnet50(**kwargs):
    return _csp_resnet(Bottleneck,[3, 4, 6, 3],**kwargs)
