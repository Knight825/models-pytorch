import torch

class SEBlock(torch.nn.Module):
    def __init__(self,inplanes,redr,poolflag = 'avg'):
        super(SEBlock,self).__init__()
        if poolflag == 'max':
            self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        if poolflag == 'avg':
            self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.replanes = inplanes//redr
        self.linear1 = torch.nn.Conv2d(inplanes,self.replanes,(1,1),padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Conv2d(self.replanes,inplanes,(1,1),padding=0)
        self.sigmod = torch.nn.Sigmoid()
    def forward(self,x):
        se = self.pool(x)
        se = self.linear1(se)
        se = self.relu(se)
        se = self.linear2(se)
        se = self.sigmod(se)
        x = se * x
        return x

class CAMBlock(torch.nn.Module):
    def __init__(self,inplanes,redr,pool = 'full'):
        super(CAMBlock,self).__init__()
        self.planes = inplanes//redr
        self.poolingavg = torch.nn.AdaptiveAvgPool2d((1,1))
        self.poolingmax = torch.nn.AdaptiveMaxPool2d((1,1))
        self.avglinear1 = torch.nn.Conv2d(inplanes,self.planes,(1,1),padding=0)
        self.maxlinear1 = torch.nn.Conv2d(inplanes,self.planes,(1,1),padding=0)
        self.relu = torch.nn.ReLU(inplace = True)
        self.avglinear2 = torch.nn.Conv2d(self.planes,inplanes,(1,1),padding=0)
        self.maxlinear2 = torch.nn.Conv2d(self.planes,inplanes,(1,1),padding=0)
        self.sigmod = torch.nn.Sigmoid()
    def forward(self,x):
        x1 = self.poolingavg(x)
        x2 = self.poolingmax(x)

        x1 = self.avglinear1(x1)
        x1 = self.relu(x1)
        x1 = self.avglinear2(x1)

        x2 = self.maxlinear1(x2)
        x2 = self.relu(x2)
        x2 = self.maxlinear2(x2)

        out = x1+x2
        out = self.sigmod(out)
        out = x*out

        return out