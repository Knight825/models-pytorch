import torch


def _Add_DifferentChannels(tensorA,tensorB):
    _,ac,_,_ = tensorA.shape
    _,bc,_,_ = tensorB.shape
    if ac == bc:
        return tensorA+tensorB

    partiralchannels,shorttensor,longtensor = ac,tensorA,tensorB

    if bc<ac:
        partiralchannels,shorttensor,longtensor = bc,tensorB,tensorA
    
    return torch.cat([shorttensor+longtensor[:,:partiralchannels,...],longtensor[:,partiralchannels:,...]],dim=1)