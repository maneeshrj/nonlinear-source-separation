import torch
import torch.nn as nn
import torch.nn.functional as F

### 3D UNet model ###


# Downsampling block
class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottleneck block
    :param batch_norm -> specifies if batch normalization is to be used
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, batch_norm = False, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)

        self.activation = nn.LeakyReLU()
        
        self.batch_norm = batch_norm
        self.bottleneck = bottleneck

        if batch_norm:
            self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
            self.bn2 = nn.BatchNorm3d(num_features=out_channels)

        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):

        if self.batch_norm:
            res = self.activation(self.bn1(self.conv1(input)))
            res = self.activation(self.bn2(self.conv2(res)))
        else:
            res = self.activation(self.conv1(input))
            res = self.activation(self.conv2(res))
        
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


# Upsampling block
class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    :param batch_norm -> specifies if batch normalization is to be used
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, batch_norm=False, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.activation = nn.LeakyReLU()

        self.bn1 = nn.BatchNorm3d(num_features=in_channels//2)
        self.bn2 = nn.BatchNorm3d(num_features=in_channels//2)

        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.batch_norm = batch_norm
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1), bias=False)
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)

        if self.batch_norm:
            out = self.activation(self.bn1(self.conv1(out)))
            out = self.activation(self.bn2(self.conv2(out)))
        else:
            out = self.activation(self.conv1(out))
            out = self.activation(self.conv2(out))
        
        if self.last_layer: out = self.conv3(out)
        return out
        

# 3D UNet architecture
class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param final_relu -> specifies if ReLU activation is to be applied at the final layer
    :param batch_norm -> specifies if batch normalization is to be used
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[16, 32, 64], bottleneck_channel=128, final_relu=False, batch_norm=False) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]

        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls, batch_norm=batch_norm)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls, batch_norm=batch_norm)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls, batch_norm=batch_norm)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, batch_norm=batch_norm, bottleneck=True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls, batch_norm=batch_norm)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls, batch_norm=batch_norm)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, batch_norm=batch_norm, last_layer=True)
        
        self.activation = nn.LeakyReLU()
        self.batch_norm = batch_norm
    
    def forward(self, input):
        # Downsampling layers
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)

        out, _ = self.bottleNeck(out)

        # Upsampling layers
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        
        return out