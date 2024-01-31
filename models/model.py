import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
import math
import numpy as np
from .layers import normalization
from .layers import general_conv3d
from .layers import prm_generator_laststage, prm_generator, region_aware_modal_fusion
from .attention import ScaledDotProductAttention

basic_dims = 16

class Encoder(nn.Module):
    def __init__(self, channel=1):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(channel, basic_dims, pad_type='reflect')
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        # feature = x4
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder, self).__init__()

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4, is_lc=False):
        super(Decoder_fuse, self).__init__()

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = region_aware_modal_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = region_aware_modal_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = region_aware_modal_fusion(in_channel=basic_dims*1, num_cls=num_cls)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_generator3 = prm_generator(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_generator2 = prm_generator(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_generator1 = prm_generator(in_channel=basic_dims*1, num_cls=num_cls)

        self.is_lc = is_lc # 是否需要插入本地校准模块
        self.x4_attn = ScaledDotProductAttention(basic_dims*8, basic_dims*8, basic_dims*8, h=8)
        self.x3_attn = ScaledDotProductAttention(basic_dims*4, basic_dims*4, basic_dims*4, h=8)
        self.x2_attn = ScaledDotProductAttention(basic_dims*2, basic_dims*2, basic_dims*2, h=8)
        self.x1_attn = ScaledDotProductAttention(basic_dims, basic_dims, basic_dims, h=4)

        self.is_gen = False

    def forward(self, x1, x2, x3, x4, mask, px1, px2, px3, px4):

        prm_pred4 = self.prm_generator4(x4, mask)   
        
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask) 
        fusion_x4= de_x4

        ####### 插入LACCA模块
        if self.is_lc:
            q_x4 = de_x4.permute(0,2,3,4,1).reshape(-1, de_x4.shape[1])    # [BNHW, C]
            att_x4, att_x4_ = self.x4_attn(q_x4, px4, px4)   #[BNHW, C]
            
            # print(att_x4.shape, att_x4_.shape, "hhhh")
   
            att_x4_ = att_x4_.reshape(de_x4.shape[0], de_x4.shape[2], de_x4.shape[3], de_x4.shape[4], -1).permute(0,4,1,2,3)
            att_x4 = att_x4.reshape(de_x4.shape[0], de_x4.shape[2], de_x4.shape[3], de_x4.shape[4], -1).permute(0,4,1,2,3)
            m_x4 = att_x4 # + de_x4
        else:
            m_x4 = de_x4
        ####################
        de_x4 = self.d3_c1(self.up2(m_x4))         
        
        if de_x4.shape[2:] != x3.shape[3:]:
            _,_,_, H, W, Z = x3.size()
            de_x4 = de_x4[:,:,:H,:W,:Z]
            
        #######################################################################
        
        
        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)
        
        de_x3 = torch.cat((de_x3, de_x4), dim=1)    
        de_x3 = self.d3_out(self.d3_c2(de_x3))

        fusion_x3 = de_x3

        ####### 插入LACCA模块
        if self.is_lc:
            q_x3 = de_x3.permute(0,2,3,4,1).reshape(-1, de_x3.shape[1])
            att_x3, att_x3_ = self.x3_attn(q_x3, px3, px3)
            att_x3 = att_x3.reshape(de_x3.shape[0], de_x3.shape[2], de_x3.shape[3], de_x3.shape[4], -1).permute(0,4,1,2,3)
            att_x3_ = att_x3_.reshape(de_x3.shape[0], de_x3.shape[2], de_x3.shape[3], de_x3.shape[4], -1).permute(0,4,1,2,3)
            m_x3 = att_x3 # + de_x3
        else:
            m_x3 = de_x3
        ####################    
        de_x3 = self.d2_c1(self.up2(m_x3))   

        if de_x3.shape[2:] != x2.shape[3:]:
            _,_,_, H, W, Z = x2.size()
            de_x3 = de_x3[:,:,:H,:W,:Z]
                
        #######################################################################
        
        
        prm_pred2 = self.prm_generator2(de_x3, x2, mask)  
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)   
        de_x2 = torch.cat((de_x2, de_x3), dim=1)       
        de_x2 = self.d2_out(self.d2_c2(de_x2))       
        fusion_x2 = de_x2

        

        ####### 插入LACCA模块
        if self.is_lc:
            q_x2 = de_x2.permute(0,2,3,4,1).reshape(-1, de_x2.shape[1])
            att_x2, att_x2_ = self.x2_attn(q_x2, px2, px2)
            # print(att_x2_.shape)
            att_x2 = att_x2.reshape(de_x2.shape[0], de_x2.shape[2], de_x2.shape[3], de_x2.shape[4], -1).permute(0,4,1,2,3)
            att_x2_ = att_x2_.reshape(de_x2.shape[0], de_x2.shape[2], de_x2.shape[3], de_x2.shape[4], -1).permute(0,4,1,2,3)
            m_x2 = att_x2 # + de_x2
        else:
            m_x2 = de_x2
        ####################
        de_x2 = self.d1_c1(self.up2(m_x2))             

        if de_x2.shape[2:] != x1.shape[3:]:
            _,_,_, H, W, Z = x1.size()
            de_x2 = de_x2[:,:,:H,:W,:Z]
            
                
        #######################################################################
        
        
        prm_pred1 = self.prm_generator1(de_x2, x1, mask)  
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)   
        de_x1 = torch.cat((de_x1, de_x2), dim=1)          
        de_x1 = self.d1_out(self.d1_c2(de_x1))        
        fusion_x1 = de_x1

        ####### 插入LACCA模块
        if self.is_lc:
            q_x1 = de_x1.permute(0,2,3,4,1).reshape(-1, de_x1.shape[1])
            att_x1, att_x1_ = self.x1_attn(q_x1, px1, px1)
            att_x1 = att_x1.reshape(de_x1.shape[0], de_x1.shape[2], de_x1.shape[3], de_x1.shape[4], -1).permute(0,4,1,2,3)
            att_x1_ = att_x1_.reshape(de_x1.shape[0], de_x1.shape[2], de_x1.shape[3], de_x1.shape[4], -1).permute(0,4,1,2,3)
            m_x1 = att_x1 # + de_x1
        else:
            m_x1 = de_x1
        
        logits = self.seg_layer(m_x1)  
        pred = self.softmax(logits)   
        
        # print(att_x1.shape, att_x1_.shape)
        
        #if self.is_gen:
        #    return (att_x1_, att_x2_, att_x3_, att_x4_)
        
        #if not self.training:
        #    return pred, (att_x1_, att_x2_, att_x3_, att_x4_)
        
        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), 
                      self.up8(prm_pred4)), (fusion_x1, fusion_x2, fusion_x3, fusion_x4)
                  


class attEDModel(nn.Module):
    def __init__(self, num_cls=4):
        super(attEDModel, self).__init__()
        # Encoder
        self.encoder = Encoder()
        # # Decoder
        # self.decoder = Decoder(num_cls=num_cls)
        # Decoder   attention
        
        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')
        
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.x4_attn = ScaledDotProductAttention(basic_dims*8, basic_dims*8, basic_dims*8, h=8)
        self.x3_attn = ScaledDotProductAttention(basic_dims*4, basic_dims*4, basic_dims*4, h=8)
        self.x2_attn = ScaledDotProductAttention(basic_dims*2, basic_dims*2, basic_dims*2, h=8)
        self.x1_attn = ScaledDotProductAttention(basic_dims, basic_dims, basic_dims, h=4)
        
        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #
        
    def forward(self, x, fx1, fx2, fx3, fx4):  
        
        # encoder
        x1, x2, x3, x4 = self.encoder(x)
        
        # decoder
        if self.is_training:
            q_x4 = x4.permute(0, 2,3,4, 1).reshape(x4.shape[0], -1, x4.shape[1])    # torch.Size([2, 1000, 128])
            kv_fx4 = fx4.repeat(x4.shape[0], 1,1)   # torch.Size([1, 4, 128])
            # kv_fx4 = fx4.repeat(x4.shape[0],1,1,1,1).permute(0, 2,3,4, 1).reshape(x4.shape[0], -1, x4.shape[1])   # torch.Size([1, 1000, 128])
            att_x4 = self.x4_attn(q_x4, kv_fx4, kv_fx4) # [B, C, N*H*W]
            att_x4 = att_x4.reshape(x4.shape)
            # att_x4_map = att_x4_map.permute(0, 2,1).reshape(x4.shape[0],-1, x4.shape[-3],x4.shape[-2],x4.shape[-1])
            # cat_x4 = torch.cat((x4, att_x4), dim=1)
            # m_x4 = self.d3_a2(self.d3_a1(cat_x4))   # torch.Size([2, 128, 10, 10, 10])
            m_x4 = att_x4 + x4
        else:
            m_x4 = x4
        
        de_x4 = self.d3_c1(self.d3(m_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        
        if self.is_training:
            q_x3 = x3.permute(0, 2,3,4, 1).reshape(x3.shape[0], -1, x3.shape[1])    # torch.Size([2, 8000, 64])
            kv_fx3 = fx3.repeat(x3.shape[0], 1,1)   # torch.Size([1, 4, 64])
            # kv_fx3 = fx3.repeat(x3.shape[0],1,1,1,1).permute(0, 2,3,4, 1).reshape(x3.shape[0], -1, x3.shape[1])   # torch.Size([2, 8000, 64])
            att_x3 = self.x3_attn(q_x3, kv_fx3, kv_fx3) # [B, C, N*H*W]
            att_x3 = att_x3.reshape(x3.shape)
            # att_x3_map = att_x3_map.permute(0, 2,1).reshape(x3.shape[0],-1, x3.shape[-3],x3.shape[-2],x3.shape[-1])
            m_x3 = att_x3 + de_x3
        else:
            m_x3 = de_x3
        
        de_x3 = self.d2_c1(self.d2(m_x3))  # torch.Size([2, 32, 40, 40, 40])

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        
        if self.is_training:
            q_x2 = x2.permute(0, 2,3,4, 1).reshape(x2.shape[0], -1, x2.shape[1])    # torch.Size([2, 8000, 64])
            kv_fx2 = fx2.repeat(x2.shape[0], 1,1)   # torch.Size([1, 4, 32])
            # kv_fx2 = fx2.repeat(x2.shape[0],1,1,1,1).permute(0, 2,3,4, 1).reshape(x2.shape[0], -1, x2.shape[1])   # torch.Size([2, 8000, 64])
            att_x2 = self.x2_attn(q_x2, kv_fx2, kv_fx2) # [B, C, N*H*W]
            att_x2 = att_x2.reshape(x2.shape)
            # att_x2_map = att_x2_map.permute(0, 2,1).reshape(x2.shape[0],-1, x2.shape[-3],x2.shape[-2],x2.shape[-1])
            m_x2 = att_x2 + de_x2
        else:
            m_x2 = de_x2
        
        de_x2 = self.d1_c1(self.d1(m_x2))  # torch.Size([2, 16, 80, 80, 80])

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1)) # torch.Size([2, 16, 80, 80, 80])

        if self.is_training:
            q_x1 = x1.permute(0, 2,3,4, 1).reshape(x1.shape[0], -1, x1.shape[1])    # torch.Size([3, 512000, 16])
            kv_fx1 = fx1.repeat(x1.shape[0], 1,1)   # torch.Size([1, 40, 16])
            # kv_fx1 = fx1.repeat(x4.shape[0],1,1,1,1).permute(0, 2,3,4, 1).reshape(x1.shape[0], -1, x1.shape[1])   # torch.Size([1, 1000, 128])
            att_x1 = self.x1_attn(q_x1, kv_fx1, kv_fx1) # [B, C, N*H*W]
            att_x1 = att_x1.reshape(x1.shape)
            # att_x1_map = att_x1_map.permute(0, 2,1).reshape(x1.shape[0],-1, x1.shape[-3],x1.shape[-2],x1.shape[-1])
            # cat_x4 = torch.cat((x4, att_x1), dim=1)
            # m_x1 = self.d1_a2(self.d1_a1(cat_x4))   # torch.Size([2, 128, 10, 10, 10])
            m_x1 = att_x1 + de_x1
        else:
            m_x1 = de_x1
        # m_x1 = de_x1
            
        logits = self.seg_layer(m_x1)
        pred = self.softmax(logits)

        return  pred, (x1, x2, x3, x4)  # (att_x1_map, att_x2_map, att_x3_map, att_x4_map)

class lcEDModel(nn.Module):
    def __init__(self, num_cls=4):
        super(lcEDModel, self).__init__()
        self.c1_ED = attEDModel(num_cls=num_cls)
        self.c2_ED = attEDModel(num_cls=num_cls)
        self.c3_ED = attEDModel(num_cls=num_cls)
        self.c4_ED = attEDModel(num_cls=num_cls)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #
        
    def forward(self, x, mask, fx1, fx2, fx3, fx4):
        preds = []
        for i in range(len(mask)):
            if mask[i]==True:
                if i==0:
                    c1_pred, _ = self.c1_ED(x[:, 0:1, :,:,:], fx1, fx2, fx3, fx4)
                    preds.append(c1_pred)
                if i==1:
                    c2_pred, _ = self.c2_ED(x[:, 1:2, :,:,:], fx1, fx2, fx3, fx4)
                    preds.append(c2_pred)
                if i==2:
                    c3_pred, _ = self.c3_ED(x[:, 2:3, :,:,:], fx1, fx2, fx3, fx4)
                    preds.append(c3_pred)
                if i==3:
                    c4_pred, _ = self.c4_ED(x[:, 3:4, :,:,:], fx1, fx2, fx3, fx4)
                    preds.append(c4_pred)    
        # c1_pred, _ = self.c1_ED(x[:, 0:1, :,:,:], fx1, fx2, fx3, fx4)
        # c2_pred, _ = self.c2_ED(x[:, 1:2, :,:,:], fx1, fx2, fx3, fx4)
        # c3_pred, _ = self.c3_ED(x[:, 2:3, :,:,:], fx1, fx2, fx3, fx4)
        # c4_pred, _ = self.c4_ED(x[:, 3:4, :,:,:], fx1, fx2, fx3, fx4)

        return  preds

class EDModel(nn.Module):
    def __init__(self, num_cls=4):
        super(EDModel, self).__init__()
        self.encoder = Encoder()

        self.decoder = Decoder(num_cls=num_cls)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #
        
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        
        # fx = torch.zeros_like(x4)
        pred = self.decoder(x1, x2, x3, x4)

        return  pred, (x1, x2, x3, x4) 

class glbEDModel(nn.Module):
    def __init__(self, num_cls=4):
        super(glbEDModel, self).__init__()

        self.encoder = Encoder(channel=4)
        self.decoder = Decoder(num_cls=num_cls)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #
        
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        
        # fx = torch.zeros_like(x4)
        pred = self.decoder(x1, x2, x3, x4)

        return  pred, (x1,x2)

class E4DModel(nn.Module):
    def __init__(self, num_cls=4):
        super(E4DModel, self).__init__()
        self.c1_encoder = Encoder()
        self.c2_encoder = Encoder()
        self.c3_encoder = Encoder()
        self.c4_encoder = Encoder()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        # self.decoder_sep = Decoder(num_cls=num_cls)
        
        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #
        
    def forward(self, x, mask):
        c1_x1, c1_x2, c1_x3, c1_x4 = self.c1_encoder(x[:,0:1,:,:,:])
        c2_x1, c2_x2, c2_x3, c2_x4 = self.c2_encoder(x[:,1:2,:,:,:])
        c3_x1, c3_x2, c3_x3, c3_x4 = self.c3_encoder(x[:,2:3,:,:,:])
        c4_x1, c4_x2, c4_x3, c4_x4 = self.c4_encoder(x[:,3:4,:,:,:])
        
        x1 = torch.stack((c1_x1, c2_x1, c3_x1, c4_x1), dim=1) #Bx4xCxHWZ   # torch.Size([1, 4, 16, 112, 112, 96])
        x2 = torch.stack((c1_x2, c2_x2, c3_x2, c4_x2), dim=1)              # torch.Size([1, 4, 32, 56, 56, 48])
        x3 = torch.stack((c1_x3, c2_x3, c3_x3, c4_x3), dim=1)
        x4 = torch.stack((c1_x4, c2_x4, c3_x4, c4_x4), dim=1)

        fuse_pred, prm_preds, fusion_preds = self.decoder_fuse(x1, x2, x3, x4, mask)

        return  fuse_pred, prm_preds, fusion_preds  # x1, x2, x3, x4, pred

class E4D4Model(nn.Module):
    def __init__(self, num_cls=4, is_lc=False):
        super(E4D4Model, self).__init__()
        self.c1_encoder = Encoder()
        self.c2_encoder = Encoder()
        self.c3_encoder = Encoder()
        self.c4_encoder = Encoder()

        self.is_gen = False
        self.decoder_fuse = Decoder_fuse(num_cls=num_cls, is_lc=is_lc)
        self.decoder_sep = Decoder(num_cls=num_cls)
        
        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #
        
    def forward(self, x, mask, fx1, fx2, fx3, fx4):
        c1_x1, c1_x2, c1_x3, c1_x4 = self.c1_encoder(x[:,0:1,:,:,:])
        c2_x1, c2_x2, c2_x3, c2_x4 = self.c2_encoder(x[:,1:2,:,:,:])
        c3_x1, c3_x2, c3_x3, c3_x4 = self.c3_encoder(x[:,2:3,:,:,:])
        c4_x1, c4_x2, c4_x3, c4_x4 = self.c4_encoder(x[:,3:4,:,:,:])
        
        x1 = torch.stack((c1_x1, c2_x1, c3_x1, c4_x1), dim=1) #Bx4xCxHWZ   # torch.Size([1, 4, 16, 80, 80, 80])
        x2 = torch.stack((c1_x2, c2_x2, c3_x2, c4_x2), dim=1)              # torch.Size([1, 4, 32, 40, 40, 40])
        x3 = torch.stack((c1_x3, c2_x3, c3_x3, c4_x3), dim=1)
        x4 = torch.stack((c1_x4, c2_x4, c3_x4, c4_x4), dim=1)
        
        if self.is_gen:
            pred, fusion_features = self.decoder_fuse(x1, x2, x3, x4, mask, fx1, fx2, fx3, fx4)
            return pred, fusion_features

        fuse_pred, prm_preds, fusion_preds = self.decoder_fuse(x1, x2, x3, x4, mask, fx1, fx2, fx3, fx4)
        
        if self.is_training:
            flair_pred = self.decoder_sep(c1_x1, c1_x2, c1_x3, c1_x4)
            t1ce_pred = self.decoder_sep(c2_x1, c2_x2, c2_x3, c2_x4)
            t1_pred = self.decoder_sep(c3_x1, c3_x2, c3_x3, c3_x4)
            t2_pred = self.decoder_sep(c4_x1, c4_x2, c4_x3, c4_x4)
            pred = torch.stack((flair_pred, t1ce_pred, t1_pred, t2_pred), dim=0)
            msk_preds = pred[mask[0], ...]
            return fuse_pred, prm_preds, fusion_preds, msk_preds

        return  fuse_pred, prm_preds, fusion_preds  # x1, x2, x3, x4, pred

if __name__ == "__main__":
    # model = EDModel(num_cls=4)  # .cuda()
    # input = torch.randn(1,1,112,112,96)
    # msk_in = torch.cat((torch.zeros(1,1,112,112,96), torch.ones(1, 1, 112,112,96)), dim=1)
    # x3, de_x4, out = model(input)
    # rmsk = F.interpolate(msk_in, [28,28,24])
    # # x = torch.randn(1,1,112,112,96)
    # from torchvision import transforms
    # import torch.nn.functional as F
    # ### 计算得到该client modality的CSA matrix
    # CSA = []
    # for c in range(4):
    #     msk = rmsk[:, c, :,:,:]    # (B,N,H,W)
    #     mF = x3 * msk   # (B,C,N,H,W)
    #     nF = de_x4 * msk    # torch.Size([1, 64, 28, 28, 24])
    #     Sc = torch.sum(msk)
    #     A_c = torch.zeros((mF.shape[1], nF.shape[1]))
    #     for m in range(mF.shape[1]):
    #         for n in range(nF.shape[1]):
    #             Fm = mF[0,m].reshape(-1)
    #             Fn = nF[0,n].reshape(-1)
    #             cos = F.cosine_similarity(Fm, Fn, dim=0)   
    #             A_c[m,n] = cos
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = E4D4Model(num_cls=4, is_lc=True).cuda()
    input = torch.randn(1,4,80,80,80).cuda()
    px1, px2, px3, px4 = torch.randn(12, 16).cuda(), torch.randn(12, 32).cuda(), torch.randn(12, 64).cuda(), torch.randn(12, 128).cuda()
    import numpy as np
    mask = torch.unsqueeze(torch.from_numpy(np.array([True, True, False, True])), dim=0)
    out = model(input, mask, px1,px2,px3,px4)
    print(out)
    # from torchsummary import summary
    # summary(model, (4,112,112,96))
    # print(model)
    
    # model = EDModel(num_cls=4)# .cuda()
    # model.is_training = True
    # input = torch.randn(2,1,80,80,80)#.cuda()
    # # Fglb = torch.randn(1,16, 80,80,80)
    # # Fglb = torch.randn(1,32, 40,40,40)
    # Fglb = torch.randn(1,64, 20,20,20)#.cuda()
    # # Fglb = torch.randn(1,128, 10,10,10)
    # outs = model(input)
    # print(outs)

    # from torchsummary import summary
    # # summary(model, (4,112,112,96))
    # # print(model)