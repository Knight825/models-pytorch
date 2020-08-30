import torch
from camblock import *
from samblock import *


class CBAMBlock(torch.nn.Module):
    def __init__(self,withCAM ,withSAM,inplanes,camredr,camflag = 'full',samsize = 7,samflag = 'full',samplanes = None):
        super(CBAMBlock,self).__init__()
        if (not withCAM) and (not withSAM):
            raise('CBAMBlock must Contain at least one of CAMblock and SAMBlock,that is mean:withCAM and withSAM cannot be False at the same time')
        self.withCAM = withCAM
        self.withSAM = withSAM
        self.cbamblock = self._maker_layer(withCAM ,withSAM,inplanes,camredr,camflag,samsize,samflag,samplanes)
    def _maker_layer(self,withCAM ,withSAM,inplanes,camredr,camflag,samsize,samflag,samplanes):
        layer = []
        if withCAM:
            CAM = None
            if camflag == 'full':
                CAM = CAMBlock
            if camflag == 'avg' or camflag == 'max':
                CAM = SEBlock
            layer.append(CAM(inplanes,camredr,camflag))
        if withSAM:
            if samplanes is not None:
                samplanes = inplanes
            layer.append(SAMblock(samsize,samflag,samplanes))
        return torch.nn.Sequential(*layer)
    
    def forward(self,x):
        out = self.cbamblock(x)
        return out