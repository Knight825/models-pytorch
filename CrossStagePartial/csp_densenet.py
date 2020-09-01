import torch
from torchvision.models.densenet import _bn_function_factory,_DenseBlock,_DenseLayer,_Transition
from collections import OrderedDict

class _Csp_Transition(torch.nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Csp_Transition, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm2d(num_input_features))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

class _Csp_DenseBlock(torch.nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False,transition = False):
        super(_Csp_DenseBlock,self).__init__()
        self.csp_num_features1 = num_input_features//2
        self.csp_num_features2 = num_input_features - self.csp_num_features1
        trans_in_features = num_layers * growth_rate
        for i in range(num_layers):
            layer = _DenseLayer(
                self.csp_num_features2 + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d'%(i+1),layer)
        self.transition = _Csp_Transition(trans_in_features,trans_in_features//2) if transition else None

    def forward(self,x):
        features = [x[:,self.csp_num_features1:,...]]
        for name,layer in self.named_children():
            if 'denselayer' in name:
                new_feature = layer(*features)
                features.append(new_feature)
        dense = torch.cat(features[1:],1)
        if self.transition is not None:
            dense = self.transition(dense)
        return torch.cat([x[:,:self.csp_num_features1,...],dense],1)


class Csp_DenseNet(torch.nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, 
                 transitionBlock = False,transitionDense = True,bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
        super(Csp_DenseNet,self).__init__()
        self.growth_down_rate = 2 if transitionBlock else 1
        self.features = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', torch.nn.BatchNorm2d(num_init_features)),
            ('relu0', torch.nn.ReLU(inplace=True)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))            
        
        num_features = num_init_features
        for i,num_layers in enumerate(block_config):
            block = _Csp_DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                transition=transitionBlock
            )
            self.features.add_module('denseblock%d'%(i+1),block)
            num_features = num_features//2 + num_layers * growth_rate // 2
            if (i != len(block_config)-1) and transitionDense:
                trans = _Transition(num_input_features=num_features,num_output_features=num_features//2)
                self.features.add_module('transition%d'%(i+1),trans)
                num_features = num_features//2
        self.features.add_module('norm5', torch.nn.BatchNorm2d(num_features))
        self.classifier = torch.nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

   

def _csp_densenet(growth_rate,block_config,num_init_features,model = 'stand',**kwargs):
    if model == 'stand':
        return Csp_DenseNet(growth_rate,block_config,num_init_features,True,True,**kwargs)
    if model == 'fusionfirst':
        return Csp_DenseNet(growth_rate,block_config,num_init_features,True,False,**kwargs)
    if model == 'fusionlast':
        return Csp_DenseNet(growth_rate,block_config,num_init_features,False,True,**kwargs)
    raise('please input right model keyword')

def csp_densenet121(growth_rate = 32,block_config = (6,12,24,16),num_init_features = 64,**kwargs):
    return _csp_densenet(growth_rate,block_config,num_init_features,**kwargs)

def csp_densenet161(growth_rate = 48,block_config = (6,12,36,24),num_init_features = 96,**kwargs):
    return _csp_densenet(growth_rate,block_config,num_init_features,**kwargs)

def csp_densenet169(growth_rate = 32,block_config = (6,12,32,32),num_init_features = 64,**kwargs):
    return _csp_densenet(growth_rate,block_config,num_init_features,**kwargs)

def csp_densenet201(growth_rate = 32,block_config = (6,12,48,32),num_init_features = 64,**kwargs):
    return _csp_densenet(growth_rate,block_config,num_init_features,**kwargs)
