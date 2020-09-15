import torch

class SAMblock(torch.nn.Module):
    def __init__(self,size = 7,model = 'full',outplanes = None):
        super(SAMblock,self).__init__()
        self.outplanes = outplanes
        if self.outplanes is None:
            self.outplanes = 1
        self.model = model
        self.conv1 = torch.nn.Conv2d(2,self.outplanes,(size,size),stride=1,padding=size//2)
        if self.model != 'full': 
            self.conv1 = torch.nn.Conv2d(1,self.outplanes,(size,size),stride=1,padding=size//2)
        self.sigmod = torch.nn.Sigmoid()

    def forward(self,x):
        if self.model == 'mean':
            meanpool = torch.mean(x,1,True)
            sp = meanpool
        if self.model == 'max':
            maxpool,_ = torch.max(x,1,True)
            sp = maxpool
        if self.model =='full':
            maxpool,_ = torch.max(x,1,True)
            meanpool = torch.mean(x,1,True)
            sp = torch.cat([maxpool,meanpool],dim=1)

        sp = self.conv1(sp)

        sp = self.sigmod(sp)
        x = sp*x
        return x
