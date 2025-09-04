import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np


class TransNonlinear(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MultiheadAttention(nn.Module):
    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64,
                 extra_nonlinear=True):
        super(MultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        self.extra_nonlinear = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(RelationUnit(feature_dim, key_feature_dim))
            if extra_nonlinear:
                self.extra_nonlinear.append(TransNonlinear(feature_dim, key_feature_dim))
            else:
                self.extra_nonlinear = None

    def forward(self, query=None, key=None, value=None,track=None
                ):
        """
        query : #pixel x batch x dim

        """
        isFirst = True
        for N in range(self.Nh):
            if(isFirst):
                if track is not None:
                    concat,track_mask,output1 = self.head[N](query, key, value,track=track)
                else:
                    concat = self.head[N](query, key, value,track=track)
                if self.extra_nonlinear:
                    concat = self.extra_nonlinear[N](concat)
                    if track is not None:
                        output1 = self.extra_nonlinear[N](output1)
                isFirst = False
            else:
                tmp = self.head[N](query, key, value,track=track)
                if self.extra_nonlinear:
                    tmp = self.extra_nonlinear[N](tmp)
                concat = torch.cat((concat, tmp), -1)

        output = concat
        if track is not None:
            return output,track_mask,output1
        else:
            return output


class RelationUnit(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.temp = 1
        self.WK = nn.Linear(feature_dim, key_feature_dim, bias=False)
        self.WQ = nn.Linear(feature_dim, key_feature_dim, bias=False)
        self.WV = nn.Linear(feature_dim, feature_dim, bias=False)
        self.trans_conv = nn.Linear(feature_dim, feature_dim, bias=False)

        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None, mask=None,track=None):
        w_k = self.WK(key)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1, 2, 0) # Batch, Dim, Len_1

        w_q = self.WQ(query)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1, 0, 2) # Batch, Len_2, Dim

        dot_prod1 = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1
        dot_prod = dot_prod1
        if mask is not None:
            dot_prod = dot_prod.masked_fill(mask == 0, -1e9)
        affinity = F.softmax(dot_prod * self.temp, dim=-1) 
        affinity = affinity / (1e-9 + affinity.sum(dim=1, keepdim=True))

        w_v = self.WV(value)
        w_v = w_v.permute(1,0,2) # Batch, Len_1, Dim
        output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
        output = output.permute(1,0,2)

        output = self.trans_conv(query - output)
        if track:
            affinity = F.softmax(dot_prod * self.temp, dim=1) 
            affinity = affinity / (1e-9 + affinity.sum(dim=-1, keepdim=True))
            affinity.permute(0,2,1)  #Batch, Len_1, Len_2
            w_v = self.WV(query)
            w_v = w_v.permute(1,0,2) # Batch, Len_2, Dim
            output1 = torch.bmm(affinity, w_v) # Batch, Len_1, Dim
            output1 = output1.permute(1,0,2)

            output1 = self.trans_conv(output1 - key)
            return F.relu(output),dot_prod1,F.relu(output1)
        else:
            return F.relu(output)
