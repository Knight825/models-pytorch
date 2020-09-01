import torch
from torchvision.models.resnet import BasicBlock,Bottleneck

class Csp_BasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        self.inplanes = inplanes
        self.idenplanes = self.idenplanes//2
        self.planes = planes
        super(Csp_BasicBlock,self).__init__(self.inplanes-self.idenplanes,self.planes,stride,downsample,
               groups,base_width,dilation,norm_layer)

    def forward(self,x):
        indentity = x[:,:self.idenplanes,...]

        out = self.conv1(x[:,self.idenplanes:,...])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            indentity = self.downsample(indentity)
        
        return torch.cat([indentity,out],1)


class Csp_Bottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        self.inplanes = inplanes
        self.idenplanes = self.idenplanes//2
        self.planes = planes
        super(Csp_Bottleneck,self).__init__(self.inplanes-self.idenplanes,self.planes,stride,downsample,
               groups,base_width,dilation,norm_layer)

    def forward(self,x):
        indentity = x[:,:self.idenplanes,...]

        out = self.conv1(x[:,self.idenplanes:,...])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            indentity = self.downsample(indentity)
        
        return torch.cat([indentity,out],1)