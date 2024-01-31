import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ScaledDotProductAttention

class Conv3D(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='reflect'):
        super(Conv3D, self).__init__()
        
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type)
        self.norm = nn.InstanceNorm3d(num_features=in_ch)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Encoder(nn.Module):
    def __init__(self, channel=1, basic_dims=8):
        super(Encoder, self).__init__()
        
        self.e1_c1 = Conv3D(channel, basic_dims, k_size=3, stride=1, padding=1)
        self.e1_c2 = Conv3D(basic_dims, basic_dims, k_size=1, stride=1, padding=0)
        self.e1_c3 = Conv3D(basic_dims, basic_dims)
        
        self.e2_c1 = Conv3D(basic_dims, basic_dims*2, k_size=5, stride=2, padding=2)
        self.e2_c2 = Conv3D(basic_dims*2, basic_dims*2, k_size=1, stride=1, padding=0)
        self.e2_c3 = Conv3D(basic_dims*2, basic_dims*2)
        
        self.e3_c1 = Conv3D(basic_dims*2, basic_dims*4, k_size=5, stride=2, padding=2)
        self.e3_c2 = Conv3D(basic_dims*4, basic_dims*4, k_size=1, stride=1, padding=0)
        self.e3_c3 = Conv3D(basic_dims*4, basic_dims*4)
        
        self.e4_c1 = Conv3D(basic_dims*4, basic_dims*8, k_size=5, stride=2, padding=2)
        self.e4_c2 = Conv3D(basic_dims*8, basic_dims*8, k_size=1, stride=1, padding=0)
        self.e4_c3 = Conv3D(basic_dims*8, basic_dims*8)
    
    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = self.e1_c3(self.e1_c2(x1))
        
        x2 = self.e2_c1(x1)
        x2 = self.e2_c3(self.e2_c2(x2))
        
        x3 = self.e3_c1(x2)
        x3 = self.e3_c3(self.e3_c2(x3))
        
        x4 = self.e4_c1(x3)
        x4 = self.e4_c3(self.e4_c2(x4))
        
        return x1, x2, x3, x4
        
    
class Decoder(nn.Module):
    def __init__(self, basic_dims=8, num_class=4, is_lc=False):
        super(Decoder, self).__init__()
        
        self.d1_c1 = Conv3D(basic_dims*8*4, basic_dims*8*4)
        self.d1_c2 = Conv3D(basic_dims*8*4, basic_dims*8*2, k_size=1, stride=1, padding=0)
        self.d1_c3 = Conv3D(basic_dims*8*2, basic_dims*8*2)
        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.s2_seg = nn.Sequential(
            nn.Conv3d(basic_dims*8*2, basic_dims*4, kernel_size=1),
            nn.Conv3d(basic_dims*4, num_class, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.d2_c = Conv3D(basic_dims*8*2*2, basic_dims*8*2)
        self.d2_c1 = Conv3D(basic_dims*8*2, basic_dims*8*2)
        self.d2_c2 = Conv3D(basic_dims*8*2, basic_dims*8, k_size=1, stride=1, padding=0)
        self.d2_c3 = Conv3D(basic_dims*8, basic_dims*8)
        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.s1_seg = nn.Sequential(
            nn.Conv3d(basic_dims*8, basic_dims*4, kernel_size=1),
            nn.Conv3d(basic_dims*4, num_class, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.d3_c = Conv3D(basic_dims*8*2, basic_dims*8)
        self.d3_c1 = Conv3D(basic_dims*8, basic_dims*8)
        self.d3_c2 = Conv3D(basic_dims*8, basic_dims*4, k_size=1, stride=1, padding=0)
        self.d3_c3 = Conv3D(basic_dims*4, basic_dims*4)
        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear')
        
        self.d4_c = Conv3D(basic_dims*4*2, basic_dims*4)
        self.d4_c1 = Conv3D(basic_dims*4, basic_dims*4)
        self.d4_c2 = Conv3D(basic_dims*4, basic_dims*2, k_size=1, stride=1, padding=0)
        self.d4_c3 = Conv3D(basic_dims*2, basic_dims*2)
        self.seg_layer = nn.Conv3d(basic_dims*2, num_class, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
        self.is_lc = is_lc 
        self.x4_attn = ScaledDotProductAttention(basic_dims*8*2, basic_dims*8*2, basic_dims*8*2, h=8)
        self.x3_attn = ScaledDotProductAttention(basic_dims*4*2, basic_dims*4*2, basic_dims*4*2, h=8)
        self.x2_attn = ScaledDotProductAttention(basic_dims*2*2, basic_dims*2*2, basic_dims*2*2, h=8)
        self.x1_attn = ScaledDotProductAttention(basic_dims*2, basic_dims*2, basic_dims*2, h=4)
        
    def forward(self, x1, x2, x3, x4, px1, px2, px3, px4):
        x1c = x1.reshape(x1.shape[0], -1, x1.shape[-3],x1.shape[-2],x1.shape[-1])  
        x2c = x2.reshape(x2.shape[0], -1, x2.shape[-3],x2.shape[-2],x2.shape[-1])  
        x3c = x3.reshape(x3.shape[0], -1, x3.shape[-3],x3.shape[-2],x3.shape[-1])  
        x4c = x4.reshape(x4.shape[0], -1, x4.shape[-3],x4.shape[-2],x4.shape[-1])  
        
        de_x4 = self.d1_c2(self.d1_c1(x4c)) 
        fusion_x4 = de_x4
        if self.is_lc:
            q_x4 = de_x4.permute(0,2,3,4,1).reshape(-1, de_x4.shape[1])   # [bnhw, c]
            att_x4 = self.x4_attn(q_x4, px4, px4)   
            att_x4 = att_x4.reshape(de_x4.shape[0], de_x4.shape[2], de_x4.shape[3], de_x4.shape[4], -1).permute(0,4,1,2,3)
            m_x4 = att_x4 + de_x4
        else:
            m_x4 = de_x4
            
        de_x3 = self.d1_c3(self.d1(m_x4))  
        s2_seg = self.s2_seg(de_x3)
        
        cat_x3 = torch.cat((de_x3, x3c), dim=1)
        de_x3 = self.d2_c2(self.d2_c1(self.d2_c(cat_x3)))   
        fusion_x3 = de_x3
        if self.is_lc:
            q_x3 = de_x3.permute(0,2,3,4,1).reshape(-1, de_x3.shape[1])   # [bnhw, c]
            att_x3 = self.x3_attn(q_x3, px3, px3)   
            att_x3 = att_x3.reshape(de_x3.shape[0], de_x3.shape[2], de_x3.shape[3], de_x3.shape[4], -1).permute(0,4,1,2,3)
            m_x3 = att_x3 + de_x3
        else:
            m_x3 = de_x3
            
        de_x2 = self.d2_c3(self.d2(m_x3))
        s1_seg = self.s1_seg(de_x2)
        
        cat_x2 = torch.cat((de_x2, x2c), dim=1)
        de_x2 = self.d3_c2(self.d3_c1(self.d3_c(cat_x2)))
        fusion_x2 = de_x2
        if self.is_lc:
            q_x2 = de_x2.permute(0,2,3,4,1).reshape(-1, de_x2.shape[1])   # [bnhw, c]
            att_x2 = self.x2_attn(q_x2, px2, px2)   
            att_x2 = att_x2.reshape(de_x2.shape[0], de_x2.shape[2], de_x2.shape[3], de_x2.shape[4], -1).permute(0,4,1,2,3)
            m_x2 = att_x2 + de_x2
        else:
            m_x2 = de_x2
        
        de_x1 = self.d3_c3(self.d3(de_x2))
        
        cat_x1 = torch.cat((de_x1, x1c), dim=1)
        de_x1 = self.d4_c2(self.d4_c1(self.d4_c(cat_x1)))
        fusion_x1 = de_x1
        if self.is_lc:
            q_x1 = de_x1.permute(0,2,3,4,1).reshape(-1, de_x1.shape[1])   # [bnhw, c]
            att_x1 = self.x1_attn(q_x1, px1, px1)   
            att_x1 = att_x1.reshape(de_x1.shape[0], de_x1.shape[2], de_x1.shape[3], de_x1.shape[4], -1).permute(0,4,1,2,3)
            m_x1 = att_x1 + de_x1
        else:
            m_x1 = de_x1
        
        de_x1 = self.seg_layer(self.d4_c3(m_x1))
        final_seg = self.softmax(de_x1)
        
        return final_seg, s2_seg, s1_seg, (fusion_x1, fusion_x2, fusion_x3, fusion_x4)


class MMmodel(nn.Module):
    def __init__(self, is_lc=False):
        super(MMmodel, self).__init__()
        
        self.c1_encoder = Encoder()
        self.c2_encoder = Encoder()
        self.c3_encoder = Encoder()
        self.c4_encoder = Encoder()
        
        self.decoder = Decoder(is_lc=is_lc)
    
    def forward(self, x, mask, px1, px2, px3, px4):
        c1_x1, c1_x2, c1_x3, c1_x4 = self.c1_encoder(x[:, 0:1, :, :, :])
        c2_x1, c2_x2, c2_x3, c2_x4 = self.c2_encoder(x[:, 1:2, :, :, :])
        c3_x1, c3_x2, c3_x3, c3_x4 = self.c3_encoder(x[:, 2:3, :, :, :])
        c4_x1, c4_x2, c4_x3, c4_x4 = self.c4_encoder(x[:, 3:4, :, :, :])
        
        x1c = torch.stack([c1_x1, c2_x1, c3_x1, c4_x1], dim=1)
        x2c = torch.stack([c1_x2, c2_x2, c3_x2, c4_x2], dim=1)
        x3c = torch.stack([c1_x3, c2_x3, c3_x3, c4_x3], dim=1)
        x4c = torch.stack([c1_x4, c2_x4, c3_x4, c4_x4], dim=1)
        
        x1, x2, x3, x4 = torch.zeros_like(x1c), torch.zeros_like(x2c), torch.zeros_like(x3c), torch.zeros_like(x4c)
        x1[:, mask], x2[:, mask], x3[:, mask], x4[:, mask] = x1c[:, mask], x2c[:, mask], x3c[:, mask], x4c[:, mask]
        
        
        pred, s2pred, s1pred, fusion_F = self.decoder(x1, x2, x3, x4, px1, px2, px3, px4)
        
        return pred, s2pred, s1pred, fusion_F


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    model = MMmodel(is_lc=True).cuda()
    
    import numpy as np
    in_x = torch.randn((2, 4, 80, 80, 80)).cuda()
    mask = torch.from_numpy(np.array([True, False, False, False]))
    px1, px2, px3, px4 = torch.randn(12, 16).cuda(), torch.randn(12, 32).cuda(), torch.randn(12, 64).cuda(), torch.randn(12, 128).cuda()
    
    out = model(in_x, mask, px1,px2,px3,px4)
    
    print(out.shape)
    