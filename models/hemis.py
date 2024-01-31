import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .attention import ScaledDotProductAttention

class Conv3D(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='reflect'):
        super(Conv3D, self).__init__()
        
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)
        self.norm = nn.InstanceNorm3d(num_features=out_ch)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Encoder(nn.Module):
    def __init__(self, channel=1, basic_dims=16) -> None:
        super(Encoder, self).__init__()
        
        self.e1_c1 = Conv3D(channel, basic_dims)
        self.e1_c2 = Conv3D(basic_dims, basic_dims, k_size=1, stride=1, padding=0)
        self.e1_c3 = Conv3D(basic_dims, basic_dims)
        
        self.e2_c1 = Conv3D(basic_dims, basic_dims*2, k_size=3, stride=2, padding=1)
        self.e2_c2 = Conv3D(basic_dims*2, basic_dims*2, k_size=1, stride=1, padding=0)
        self.e2_c3 = Conv3D(basic_dims*2, basic_dims*2)
        
        self.e3_c1 = Conv3D(basic_dims*2, basic_dims*4, k_size=3, stride=2, padding=1)
        self.e3_c2 = Conv3D(basic_dims*4, basic_dims*4, k_size=1, stride=1, padding=0)
        self.e3_c3 = Conv3D(basic_dims*4, basic_dims*4)
        
        self.e4_c1 = Conv3D(basic_dims*4, basic_dims*8, k_size=3, stride=2, padding=1)
        self.e4_c2 = Conv3D(basic_dims*8, basic_dims*8, k_size=1, stride=1, padding=0)
        self.e4_c3 = Conv3D(basic_dims*8, basic_dims*8)
        
    def forward(self, x):   
        x1 = self.e1_c1(x)  
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))
        
        return (x1,x2,x3,x4)

class Decoder(nn.Module):
    def __init__(self, num_cls=4, basic_dims=16, is_lc=False):
        super(Decoder, self).__init__()

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = Conv3D(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = Conv3D(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = Conv3D(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = Conv3D(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = Conv3D(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = Conv3D(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = Conv3D(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = Conv3D(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = Conv3D(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)
        
        self.is_lc = is_lc
        if is_lc:
            self.x4_attn = ScaledDotProductAttention(basic_dims*8, basic_dims*8, basic_dims*8, h=8)
            self.x3_attn = ScaledDotProductAttention(basic_dims*4, basic_dims*4, basic_dims*4, h=8)
            self.x2_attn = ScaledDotProductAttention(basic_dims*2, basic_dims*2, basic_dims*2, h=8)
            self.x1_attn = ScaledDotProductAttention(basic_dims, basic_dims, basic_dims, h=4)

    def forward(self, x1, x2, x3, x4, px1=None, px2=None, px3=None, px4=None):
        fusion_x4 = x4
        if self.is_lc:
            q_x4 = x4.permute(0,2,3,4,1).reshape(-1, x4.shape[1])   # [bnhw, c]
            att_x4 = self.x4_attn(q_x4, px4, px4)   
            att_x4 = att_x4.reshape(x4.shape[0], x4.shape[2], x4.shape[3], x4.shape[4], -1).permute(0,4,1,2,3)
            m_x4 = att_x4 + x4
        else:
            m_x4 = x4
        de_x4 = self.d3_c1(self.d3(m_x4))
        
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        
        fusion_x3 = de_x3
        if self.is_lc:
            q_x3 = de_x3.permute(0,2,3,4,1).reshape(-1, de_x3.shape[1]) # [bnhw, c]
            att_x3 = self.x3_attn(q_x3, px3, px3)
            att_x3 = att_x3.reshape(de_x3.shape[0], de_x3.shape[2], de_x3.shape[3], de_x3.shape[4], -1).permute(0,4,1,2,3)
            m_x3 = att_x3 + de_x3
        else:
            m_x3 = x3   
        de_x3 = self.d2_c1(self.d2(m_x3))
        
        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        
        fusion_x2 = de_x2
        if self.is_lc:
            q_x2 = de_x2.permute(0, 2,3,4, 1).reshape(-1, de_x2.shape[1])    
            att_x2 = self.x2_attn(q_x2, px2, px2)
            att_x2 = att_x2.reshape(de_x2.shape[0], de_x2.shape[2], de_x2.shape[3], de_x2.shape[4], -1).permute(0,4,1,2,3)
            m_x2 = att_x2 + de_x2
        else:
            m_x2 = de_x2
        de_x2 = self.d1_c1(self.d1(m_x2))
        
        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))
        
        fusion_x1 = de_x1
        if self.is_lc:
            q_x1 = de_x1.permute(0, 2,3,4, 1).reshape(-1, de_x1.shape[1])
            att_x1 = self.x1_attn(q_x1, px1, px1)
            att_x1 = att_x1.reshape(de_x1.shape[0], de_x1.shape[2], de_x1.shape[3], de_x1.shape[4], -1).permute(0,4,1,2,3)
            m_x1 = att_x1 + de_x1
        else:
            m_x1 = de_x1
        logits = self.seg_layer(m_x1)
        pred = self.softmax(logits)
            
        return pred, (fusion_x1, fusion_x2, fusion_x3, fusion_x4)
    
class Combine_Conv(nn.Module):
    def __init__(self, dims=128):
        super(Combine_Conv, self).__init__()
        self.module = nn.Sequential(
            nn.Conv3d(dims*2, dims, kernel_size=1, stride=1, padding=0),
            Conv3D(dims, dims, k_size=3, stride=1, padding=1),
            Conv3D(dims, dims, k_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        
        x = self.module(x)
        
        return x            
        
class Unet3D_HeMIS(nn.Module):
    def __init__(self, basic_dims=16, is_lc=False):
        super(Unet3D_HeMIS, self).__init__()
        
        self.branches = nn.ModuleDict()
        self.combines = nn.ModuleDict()
        
        n_modals = 4
        for modal in range(n_modals):
            name = 'modal_{}'.format(modal)
            if modal == 0:
            
                self.c1_encoder = Encoder()
                self.branches[name] = self.c1_encoder
            if modal == 1:
            
                self.c2_encoder = Encoder()
                self.branches[name] = self.c2_encoder
            if modal == 2:
            
                self.c3_encoder = Encoder()
                self.branches[name] = self.c3_encoder
            if modal == 3:
            
                self.c4_encoder = Encoder()
                self.branches[name] = self.c4_encoder
                
            scale = str(modal)
            combined = Combine_Conv(dims=16*(2**modal))
            self.combines[scale] = combined
        
        self.decoder = Decoder(is_lc=is_lc)
        # self.c1_encoder = Encoder()
        # self.c2_encoder = Encoder()
        # self.c3_encoder = Encoder()
        # self.c4_encoder = Encoder()
        
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.constant_(m.bias, 0)
    
    def forward(self, images, mask, px1,px2,px3,px4): 
        features = []          
        branch_names = list(self.branches.keys())
        
        for idx in range(images.shape[1]):
            modal_image = images[:, idx:idx + 1]
            name = branch_names[idx]
            modal_features = self.branches[name](modal_image)
            features.append(modal_features)
        
        fuse_Fs = []
        for scale in range(4):
            sc_features = []
            for m_features in features:
                sc_features.append(m_features[scale])   
            scale_features = torch.stack(sc_features, dim=1)    # [b, 4, c, n,h,w]
            msk_features = scale_features[:, mask[0]]
            if msk_features.shape[1] == 1:
                mean_var = [msk_features[:,0], torch.zeros_like(msk_features[:,0])]
            else:
                mean_var = [torch.mean(msk_features, dim=1), torch.var(msk_features, dim=1)]
            cat_features = torch.cat(mean_var, dim=1)      # [b, 2*c, n,h,w]
            fusion_features = self.combines[str(scale)](cat_features)   
            fuse_Fs.append(fusion_features)
        
        pred, fusion_F = self.decoder(fuse_Fs[0], fuse_Fs[1], fuse_Fs[2], fuse_Fs[3],
                                      px1, px2, px3, px4)
        
        if self.training:
            return pred, [], fusion_F, None
        else:
            return pred, [], fusion_F
            
class Unet3D_HeMIS_1e1d(nn.Module):
    def __init__(self, basic_dims=16, is_lc=False, inc = 1):
        super(Unet3D_HeMIS, self).__init__()
        
        self.encoder = Encoder(channel = inc)
             
        
        self.decoder = Decoder(is_lc=False)
        # self.c1_encoder = Encoder()
        # self.c2_encoder = Encoder()
        # self.c3_encoder = Encoder()
        # self.c4_encoder = Encoder()
        
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.constant_(m.bias, 0)
    
    def forward(self, images, mask, px1,px2,px3,px4):
        fuse_Fs = self.encoder(images[:, mask[0]])
        
        
        pred, fusion_F = self.decoder(fuse_Fs[0], fuse_Fs[1], fuse_Fs[2], fuse_Fs[3])
        
        if self.training:
            return pred, [], fusion_F, None
        else:
            return pred, [], fusion_F
    

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    model = Unet3D_HeMIS(is_lc=True).cuda()
    
    input = torch.randn(4, 1, 1, 80,80,80).cuda()
    mask = torch.from_numpy(np.array([True, False, False, False]))
    px1, px2, px3, px4 = torch.randn(12, 16).cuda(), torch.randn(12, 32).cuda(), torch.randn(12, 64).cuda(), torch.randn(12, 128).cuda()
    
    out, _ = model(input, mask, px1,px2,px3,px4)
    
    print(out.shape)