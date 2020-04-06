import torch
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Attention3d(nn.Module):
    def __init__(self, channels, reduction_attn=8, reduction_sc=2):
        super().__init__()
        self.channles_attn = channels // reduction_attn
        self.channels_sc = channels // reduction_sc
        self.maxpooling3d = nn.MaxPool3d(2,2)
        self.conv_query = spectral_norm(nn.Conv3d(channels, self.channles_attn, kernel_size=1, bias=False))
        self.conv_key = spectral_norm(nn.Conv3d(channels, self.channles_attn, kernel_size=1, bias=False))
        self.conv_value = spectral_norm(nn.Conv3d(channels, self.channels_sc, kernel_size=1, bias=False))
        self.conv_attn = spectral_norm(nn.Conv3d(self.channels_sc, channels, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        
        nn.init.orthogonal_(self.conv_query.weight.data)
        nn.init.orthogonal_(self.conv_key.weight.data)
        nn.init.orthogonal_(self.conv_value.weight.data)
        nn.init.orthogonal_(self.conv_attn.weight.data)

    def forward(self, x):
        batch, _,l, h, w = x.size()
        
        proj_query = self.conv_query(x).view(batch, self.channles_attn, -1)
        proj_key = self.maxpooling3d(self.conv_key(x)).view(batch, self.channles_attn, -1)
        
        attn = torch.bmm(proj_key.permute(0,2,1), proj_query)
        attn = F.softmax(attn, dim=1)
        
        proj_value = self.maxpooling3d(self.conv_value(x)).view(batch, self.channels_sc, -1)
        attn = torch.bmm(proj_value, attn)
        attn = attn.view(batch, self.channels_sc,l, h, w)
        attn = self.conv_attn(attn)
        
        out = self.gamma * attn + x
        
        return out



def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    nn.init.orthogonal_(conv_layer.weight.data)
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))
    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):

    layers = []
    de_layer = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    nn.init.orthogonal_(de_layer.weight.data)
    layers.append(de_layer)
    if batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                                kernel_size=3, stride=1, padding=1, batch_norm=True)
        
        self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                               kernel_size=3, stride=1, padding=1, batch_norm=True)
        
    def forward(self, x):
        out_1 = F.relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1)
        return out_2

def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules: print("{}: {}".format(n, num_params))
        total_num_params += num_params
    print("-" * 50)
    print("Total number of parameters: {:.2e}".format(total_num_params))
    



class Effinet(nn.Module):
    def __init__(self,Ch=16,n_res_blocks=4):
        super(Effinet,self).__init__()   
        self.block1 = conv(1,Ch//2,4)
        self.pooling = nn.MaxPool3d(2)
        self.attn1 = Attention3d(Ch//2)
        self.attn2 = Attention3d(Ch//2)
        self.attn3 = Attention3d(Ch//2)
        self.attn4 = Attention3d(Ch//2)
        self.attn5 = Attention3d(Ch//2)
        self.attn6 = Attention3d(Ch//2)
        self.attn7 = Attention3d(Ch//2)
        self.attn8 = Attention3d(Ch//2)

        self.conv1 = conv(Ch*4,Ch*6,4)
        self.conv2 = conv(Ch*6, Ch*8,4)
        self.deconv1 = deconv(Ch*8, Ch*4, 4)
        self.deconv2 = deconv(Ch*4, Ch, 4)
        self.deconv3 = deconv(Ch, 1, 4, batch_norm=False)

        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(Ch*8))
        self.res_blocks = nn.Sequential(*res_layers)
        self.metrics_keys = ['loss', 'mse','mae']

    def forward(self,x):
        x = self.block1(x)
        x1 = self.attn1(x)
        x2 = self.attn2(x)
        x3 = self.attn3(x)
        x4 = self.attn4(x)
        x5 = self.attn5(x)
        x6 = self.attn6(x)
        x7 = self.attn7(x)
        x8 = self.attn8(x)
        x = torch.cat([x1,x2,x3,x4,x5,x6,x7,x8],dim=1)
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res_blocks(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x # 32, 64, 32


def get_conv_inorm_relu(in_planes, out_planes, kernel_size, stride, reflection_pad=True, with_relu=True):
    layers = []
    padding = (kernel_size - 1) // 2
    if reflection_pad:
        layers.append(nn.ReplicationPad3d(padding=padding))
        padding = 0
    layers += [
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.InstanceNorm3d(out_planes, affine=False, track_running_stats=False),
    ]
    if with_relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def get_conv_transposed_inorm_relu(in_planes, out_planes, kernel_size, stride):
    return nn.Sequential(
        nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1),
        nn.InstanceNorm3d(out_planes, affine=False, track_running_stats=False),
        nn.ReLU(inplace=True)
    )

class ResiBlock(nn.Module):
    
    def __init__(self, in_planes):
        super(ResiBlock, self).__init__()
        self.conv1 = get_conv_inorm_relu(in_planes, in_planes, kernel_size=3, stride=1)
        self.conv2 = get_conv_inorm_relu(in_planes, in_planes, kernel_size=3, stride=1, with_relu=False)        

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)        
        return x + residual

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.c7s1_64 = get_conv_inorm_relu(1, 64, kernel_size=7, stride=1)
        self.d128 = get_conv_inorm_relu(64, 128, kernel_size=3, stride=2, reflection_pad=False)
        self.d256 = get_conv_inorm_relu(128, 256, kernel_size=3, stride=2, reflection_pad=False)

        self.resnet9 = nn.Sequential(*[ResidualBlock(256) for i in range(9)])

        self.u128 = get_conv_transposed_inorm_relu(256, 128, kernel_size=3, stride=2)
        self.u64 = get_conv_transposed_inorm_relu(128, 64, kernel_size=3, stride=2)
        self.c7s1_3 = get_conv_inorm_relu(64, 1, kernel_size=7, stride=1, with_relu=False)
        # Replace instance norm by tanh activation
        self.c7s1_3[-1] = nn.Tanh()

    def forward(self, x):
        # Encoding
        x = self.c7s1_64(x)
        x = self.d128(x)
        x = self.d256(x)
        
        # 9 residual blocks
        # print(x.size())
        x = self.resnet9(x)

        # Decoding
        x = self.u128(x)
        x = self.u64(x)
        # print(x.size())
        y = self.c7s1_3(x)
        return y
    
def get_conv_inorm_lrelu(in_planes, out_planes, stride=2, negative_slope=0.2):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1),
        nn.InstanceNorm3d(out_planes, affine=False, track_running_stats=False),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    )


class Discriminators(nn.Module):

    def __init__(self):
        super(Discriminators, self).__init__()
        self.c64 = nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.c128 = get_conv_inorm_lrelu(64, 128)
        self.c256 = get_conv_inorm_lrelu(128, 256)
        self.c512 = get_conv_inorm_lrelu(256, 512, stride=1)
        self.last_conv = nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.c64(x)
        x = self.relu(x)

        x = self.c128(x)
        x = self.c256(x)
        x = self.c512(x)
        y = self.last_conv(x)
        return y # torch.Size([1, 1, 2, 6, 2])


if __name__ == "__main__":
    net = Effinet()
    print_num_params(net)
