import torch
from collections import OrderedDict

def _conv2d(in_channels,out_channels,size = 1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels,out_channels,size,stride=1,padding=size//2),
        torch.nn.BatchNorm2d(out_channels)
        )


def _downsample(in_channels,out_channels,size = 1):
    if size != 1 and size != 3:
        raise('size must be 1 or 3')
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels,out_channels,size, stride = 2,padding = size//2),
        torch.nn.BatchNorm2d(out_channels)
    )


#from bottom to top
class ExactFusionModel(torch.nn.Module):
    def __init__(self,in_clannels_list,out_channels,transition = 128,withproduction = True):
        if len(in_clannels_list) < 4:
            raise('lenght of in_channels_list must be longer than 3')
        super(ExactFusionModel,self).__init__()
        self.in_clannels_list = in_clannels_list
        self.same_blocks = torch.nn.ModuleList()
        #self.down_blocks = torch.nn.ModuleList()
        self.prod_blocks = torch.nn.ModuleList()
        self.upto_blocks = torch.nn.ModuleList()
        
        b_index = len(in_clannels_list) - 1
        up_channel = self.in_clannels_list[b_index] + self.in_clannels_list[b_index-1] - self.in_clannels_list[b_index-1]//2 
        self.efm_channels = [up_channel]
        
        b_index -= 1
        while b_index > 0:
            channels = self.in_clannels_list[b_index]//2 + self.in_clannels_list[b_index - 1] - self.in_clannels_list[b_index - 1]//2 + up_channel if transition < 1 else self.in_clannels_list[b_index]//2 + self.in_clannels_list[b_index - 1] - self.in_clannels_list[b_index - 1]//2 + transition
            up_channel = channels
            self.efm_channels.insert(0,channels)
            b_index -= 1
        

        for in_channel in self.efm_channels:
            self.same_blocks.append(_conv2d(in_channel,out_channels,3))
            #self.down_blocks.append(_downsample(out_channels,out_channels,1))
            self.prod_blocks.append(_conv2d(out_channels,out_channels,3) if withproduction else torch.nn.Identity())
            self.upto_blocks.append(torch.nn.Identity() if transition > 0 else _conv2d(in_channel,transition,1))
        for m in self.children():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=1)
                torch.nn.init.constant_(m.bias, 0)

    #input must be dict,from bottom to top
    def forward(self,x):
        names = list(x.keys())
        x = list(x.values())

        xb_index = len(x) - 1
        shape = x[xb_index].shape[-2:]
        csp_x = [torch.cat([torch.nn.functional.interpolate(x[xb_index - 1][:,self.in_clannels_list[xb_index - 1]//2:,...],size=shape, mode="nearest"),x[xb_index]],1)]
        xb_index -= 1
        
        while xb_index > 0:
            shape = x[xb_index].shape[-2:]
            csp_x.insert(0,torch.cat([
                        torch.nn.functional.interpolate(x[xb_index][:,self.in_clannels_list[xb_index - 1]//2:,...],size=shape,mode='nearest'),
                        x[xb_index][:,:self.in_clannels_list[xb_index]//2,...],
                        torch.nn.functional.interpolate(self.upto_blocks[xb_index - 1](csp_x[0]),size=shape,mode='nearest')],1))
            xb_index -= 1
        
        bottom_feature = self.same_blocks[0](csp_x[0])
        result = [self.prod_blocks[0](bottom_feature)]
        for csp,same_block,prod_block in zip([csp_x[1:],self.same_blocks[1:],self.prod_blocks[1:]]):
            shape = csp.shape[-2:]
            feature = same_block(csp)
            feature = feature + torch.nn.functional.interpolate(bottom_feature,size=shape,mode='nearest')
            bottom_feature = feature
            result.append(prod_block(feature))
        
        out = OrderedDict((k,v) for k ,v in zip(names[1:],result))

        return out
