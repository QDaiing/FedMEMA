import numpy as np
import torch
from torch import nn
from torch.nn import init

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        # self.fc_k = nn.Linear(d_k, h * d_k)
        # self.fc_v = nn.Linear(d_v, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1X1 = nn.Conv2d(h, 1, kernel_size=1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        # queries # B*N*H*W, C
        # keys # k*cls, C
        # values # k*cls, C
        nq = queries.shape[0] 
        nk = keys.shape[0] 

        q = self.fc_q(queries).view(nq, self.h, self.d_k).permute(1, 0, 2)  # (h, nq, d_k)

        k = self.fc_k(keys).view(nk, self.h, self.d_k).permute(1, 2, 0)  # (h, d_k, nk)

        v = self.fc_v(values).view(nk, self.h, self.d_v).permute(1, 0, 2)  # (h, nk, d_v)

        
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (h, nq, nk)    [h, B*N*H*W, k*cls]

        att_map = torch.mean(att, 0) 
        #att_map = att_map.reshape(att_map.shape[0], 3, 4).mean(1)
        #att_map = torch.softmax(att_map.squeeze(1), -1)  

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)   # torch.Size([8, 1000, 4])

        out = torch.matmul(att, v).permute(1, 0, 2).contiguous().view(nq, self.h * self.d_v)  # (nq, h*d_v)
        # torch.matmul(att, v).shape - [h, B*N*H*W, C]
        # permute - [B*N*H*W, h, C]
        # view - [B*N*H*W, h*C]
        out = self.fc_o(out)  # [B*N*H*W, C]
        # out = out.permute(1, 0)
        return out, att_map   


if __name__ == '__main__':

    q = torch.randn([2, 128, 10,10,10 ])
    # c = q.shape[1]
    
    q = q.permute(0,2,3,4,1)
    
    q = q.reshape(q.shape[0], -1, q.shape[-1])
    
    sa = ScaledDotProductAttention(d_model=128, d_k=128, d_v=128, h=4)
    
    output=sa(q, q, q)
    print(output.shape)