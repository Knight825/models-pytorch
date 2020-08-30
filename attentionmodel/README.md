CBAM module contains two part SAM(spatial attention module) and CAM(channel attention module)

In the program,you can select the model of attention module,include:only CAM,only SAM or CBAM.
when you select only the CAM,you will get a CNN like SENet.

About CABM_ResNet or SE_ResNet,there are four style module for BasicBlock(used in resnet18 and resnet34) and Bottleneck(used in resnet50 and resnet101),
include  Standard block,Pre block,Post block and Identity block.(Paper:Squeeze-and-Excitation Networks:http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)
You can build your resnet with normal Residual module and Residual module with attention module in the same time by select the layers.
