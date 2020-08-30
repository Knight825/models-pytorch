import torch
import torchvision
from torchvision.models.resnet import BasicBlock,Bottleneck,conv1x1,ResNet
from camblock import SEBlock,CAMBlock
from samblock import SAMblock
from cbamblock import CBAMBlock


class StandCBAM_BasicBlock(BasicBlock):
    expansion = 1
    def __init__(self,inplanes,planes,withCAM,withSAM,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,redr=16,camflag = 'full',
                 samsize = 7,samflag = 'full',samplanes = None):
        super(StandCBAM_BasicBlock,self).__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer)
        self.cbamblock = CBAMBlock(withCAM,withSAM,self.conv2.out_channels,redr,camflag,samsize,samflag,samplanes)
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.cbamblock(out)

        out += identity
        out = self.relu(out)
        
        return out

class PostCBAM_BasicBlock(StandCBAM_BasicBlock):
    expansion = 1
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity

        out = self.cbamblock(out)
        out = self.relu(out)

        return out

class PreCBAM_BasicBlock(BasicBlock):
    expansion = 1
    def __init__(self,inplanes,planes,withCAM,withSAM,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,redr=16,camflag = 'full',
                 samsize = 7,samflag = 'full',samplanes = None):
        super(PreCBAM_BasicBlock,self).__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer)
        self.cbamblock = CBAMBlock(withCAM,withSAM,inplanes,redr,camflag,samsize,samflag,samplanes)
    def forward(self,x):
        identity = x

        out = self.cbamblock(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  

        return out    

class IdenCBAM_BasicBlock(PreCBAM_BasicBlock):
    expansion = 1
    def forward(self,x):
        identity = self.cbamblock(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity

        return out

class StandCBAM_Bottleneck(Bottleneck):
    expansion = 4
    def __init__(self,inplanes,planes,withCAM,withSAM,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,redr=16,camflag = 'full',
                 samsize = 7,samflag = 'full',samplanes = None):
        super(StandCBAM_Bottleneck,self).__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer)
        self.cbamblock = CBAMBlock(withCAM,withSAM,self.conv3.out_channels,redr,camflag,samsize,samflag,samplanes)
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.cbamblock(out)

        out += identity
        out = self.relu(out)
        
        return out

class PostCBAM_Bottleneck(StandCBAM_Bottleneck):
    expansion = 4
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity

        out = self.cbamblock(out)
        out = self.relu(out)

        return out

class PreCBAM_Bottleneck(Bottleneck):
    expansion = 4
    def __init__(self,inplanes,planes,withCAM,withSAM,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,redr=16,camflag = 'full',
                 samsize = 7,samflag = 'full',samplanes = None):
        super(PreCBAM_Bottleneck,self).__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer)
        self.cbamblock = CBAMBlock(withCAM,withSAM,inplanes,redr,camflag,samsize,samflag,samplanes)
    def forward(self,x):
        identity = x

        out = self.cbamblock(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  

        return out    

class IdenCBAM_Bottleneck(PreCBAM_Bottleneck):
    expansion = 4
    def forward(self,x):
        identity = self.cbamblock(x)

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
        
        out += identity

        return out


class CBAM_ResNet(torch.nn.Module):
    def __init__(self,block,cbamblock,layers,withCamlayers,withSamlayers,reductions,
                    camflag = 'full',samsize = 7,samflag = 'full',samplanes = None,
                    num_classes=1000, zero_init_residual=False,groups=1, 
                    width_per_group=64, replace_stride_with_dilation=None,
                    norm_layer=None):
        super(CBAM_ResNet,self).__init__()

        if block.expansion != cbamblock.expansion:
            raise("block and seblock have different expansion")
        
        self.cbamlayer = []

        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
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

        self.layer1 = self._make_layer(block,cbamblock,64,layers[0],withCamlayers[0],withSamlayers[0],redr=reductions[0],
                                       camflag=camflag,samsize=samsize,samflag=samflag,samplanes=samplanes)
        self.layer2 = self._make_layer(block,cbamblock,128,layers[1],withCamlayers[1],withSamlayers[1],stride=2,
                                       dilate=replace_stride_with_dilation[0],redr=reductions[1],camflag=camflag,
                                       samsize=samsize,samflag=samflag,samplanes=samplanes)
        self.layer3 = self._make_layer(block,cbamblock,256,layers[2],withCamlayers[2],withSamlayers[2],stride=2,
                                       dilate=replace_stride_with_dilation[1],redr=reductions[2],camflag=camflag,
                                       samsize=samsize,samflag=samflag,samplanes=samplanes)
        self.layer4 = self._make_layer(block,cbamblock,512,layers[3],withCamlayers[3],withSamlayers[3],stride=2,
                                       dilate=replace_stride_with_dilation[2],redr=reductions[3],camflag=camflag,
                                       samsize=samsize,samflag=samflag,samplanes=samplanes)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)    

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)    
    
    def _make_layer(self,block,cbamblock,planes,blocks,withCAM,withSAM,stride=1, dilate=False,redr = 16,
                    camflag = 'full',samsize = 7,samflag = 'full',samplanes = None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layer = []


        if ((not withCAM) and (not withSAM)):
            layer.append(block(self.inplanes, planes, stride, downsample, self.groups,
                               self.base_width, previous_dilation, norm_layer))
        else:
            _block = cbamblock(self.inplanes,planes,withCAM,withSAM,stride, downsample,self.groups,
                               self.base_width, previous_dilation, norm_layer,redr,camflag,samsize,
                               samflag,samplanes)
            layer.append(_block)
            self.cbamlayer.append(_block.cbamblock)
        
        self.inplanes = planes * block.expansion
        for _ in range(1,blocks):
            if ((not withCAM) and (not withSAM)):
                layer.append(block(self.inplanes, planes, groups=self.groups,
                             base_width=self.base_width, dilation=self.dilation,
                             norm_layer=norm_layer))
            else:
                _block = cbamblock(self.inplanes,planes,withCAM,withSAM,groups=self.groups,base_width=self.base_width, 
                                   dilation=self.dilation, norm_layer = norm_layer,redr = redr,camflag = camflag,
                                   samsize=samsize,samflag = samflag,samplanes=samplanes)
                layer.append(_block)
                self.cbamlayer.append(_block.cbamblock)

        return torch.nn.Sequential(*layer)  
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


def BasicBlockWeightsCopy(dstmodel,srcmodel):
    with torch.no_grad():
        dstmodel.conv1.weight = srcmodel.conv1.weight
        dstmodel.conv1.bias = srcmodel.conv1.bias
        dstmodel.bn1.weight = srcmodel.bn1.weight
        dstmodel.bn1.bias = srcmodel.bn1.bias
        layers = [[dstmodel.layer1,srcmodel.layer1],[dstmodel.layer2,srcmodel.layer2],[dstmodel.layer3,srcmodel.layer3],[dstmodel.layer4,srcmodel.layer4]]
        for layer in layers:
            mlayer = layer[0]
            rlayer = layer[1]
            for index,m in enumerate(mlayer):
                m.conv1.weight = rlayer[index].conv1.weight
                m.conv2.weight = rlayer[index].conv2.weight
                m.conv1.bias = rlayer[index].conv1.bias
                m.conv2.bias = rlayer[index].conv2.bias
                m.bn1.weight = rlayer[index].bn1.weight
                m.bn2.weight = rlayer[index].bn2.weight
                m.bn1.bias = rlayer[index].bn1.bias
                m.bn2.bias = rlayer[index].bn2.bias

def BottleneckWeightsCopy(dstmodel,srcmodel):
    with torch.no_grad():
        dstmodel.conv1.weight = srcmodel.conv1.weight
        dstmodel.conv1.bias = srcmodel.conv1.bias
        dstmodel.bn1.weight = srcmodel.bn1.weight
        dstmodel.bn1.bias = srcmodel.bn1.bias
        layers = [[dstmodel.layer1,srcmodel.layer1],[dstmodel.layer2,srcmodel.layer2],[dstmodel.layer3,srcmodel.layer3],[dstmodel.layer4,srcmodel.layer4]]
        for layer in layers:
            mlayer = layer[0]
            rlayer = layer[1]
            for index,m in enumerate(mlayer):
                m.conv1.weight = rlayer[index].conv1.weight
                m.conv2.weight = rlayer[index].conv2.weight
                m.conv3.weight = rlayer[index].conv3.weight
                m.conv1.bias = rlayer[index].conv1.bias
                m.conv2.bias = rlayer[index].conv2.bias
                m.conv3.bias = rlayer[index].conv3.bias
                m.bn1.weight = rlayer[index].bn1.weight
                m.bn2.weight = rlayer[index].bn2.weight
                m.bn3.weight = rlayer[index].bn3.weight
                m.bn1.bias = rlayer[index].bn1.bias
                m.bn2.bias = rlayer[index].bn2.bias
                m.bn3.bias = rlayer[index].bn3.bias

def _cbam_resnet(block,cbamblock,layers,withCamlayers,withSamlayers,reductions,**kwargs):
    model = CBAM_ResNet(block,cbamblock,layers,withCamlayers,withSamlayers,reductions,**kwargs)
    return model

def cbam_resnet50(cbamblock = StandCBAM_Bottleneck,layers = [3,4,6,3],withCamlayers=[True,True,True,True],withSamlayers=[True,True,True,True],reductions=[16, 32, 64, 128],weightspath=None,**kwargs):
    model = _cbam_resnet(Bottleneck,cbamblock,layers,withCamlayers,withSamlayers,reductions,**kwargs)
    if weightspath is not None:
        resnet50 = torchvision.models.resnet50()
        resnet50.load_state_dict(torch.load(weightspath))
        BottleneckWeightsCopy(model,resnet50)
    return model

def cbam_resnet50_layer4(withCamlayers=[False,False,False,True],withSamlayers=[False,False,False,True],**kwargs):
    model = cbam_resnet50(withCamlayers=withCamlayers,withSamlayers=withSamlayers)
    return model

def se_resnet50(**kwargs):
    model = cbam_resnet50(withSamlayers=[False,False,False,False],**kwargs)
    return model

def se_resnet50(**kwargs):
    model = cbam_resnet50_layer4(withSamlayers=[False,False,False,False],**kwargs)
    return model
