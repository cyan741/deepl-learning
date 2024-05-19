import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import dgl.function as fn
import dgl
class GCNLayer(nn.Module):
    def __init__(self,in_feats,out_feats,bias=True):
        super(GCNLayer,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats,out_feats))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_feats))
        else:
            self.bias = None

        self.reset_parameter()
        
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self,g,h):
        with g.local_scope():
            # H * W
            h = torch.matmul(h,self.weight)
            # D^{-1/2} * H * W
            g.ndata['h'] = h * g.ndata['norm']
            #对应的是AH的计算，即将每个节点的特征通过邻接关系传播给邻居节点，相当于邻接矩阵与节点特征矩阵相乘
            # A * D^{-1/2} * H * W
            g.update_all(message_func = fn.copy_u('h','m'),
                            reduce_func=fn.sum('m','h'))
            h = g.ndata['h']
            # D^{-1/2} * A * D^{-1/2} * H * W
            h = h * g.ndata['norm']
            if self.bias is not None:
                h = h + self.bias
            return h
        
class PairNorm(nn.Module):
    def __init__(self, scale=1):
        super(PairNorm, self).__init__()
        self.scale = scale

    def forward(self, x):
        col_mean = x.mean(dim=0)
        rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()# 范数
        x = self.scale * x / rownorm_individual - col_mean# 归一化后减去均值
        return x
    
class GCNModel_for_NC (nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, bias, pair_norm, activation, dropedge=False, p=0):
        super(GCNModel_for_NC,self).__init__()
        self.bias = bias
        self.pair_norm = pair_norm
        self.num_layers = num_layers
        self.activation = activation
        self.dropedge = dropedge
        self.p = p
        self.conv1 = GCNLayer(in_feats,h_feats,bias)
        self.conv2 = GCNLayer(h_feats,h_feats,bias)
        self.conv3 = GCNLayer(h_feats,num_classes,bias)
    
    def forward(self, g, in_feat):
        if self.dropedge:
            g = dgl.transforms.DropEdge(p=self.p)(g)
        for i in range(self.num_layers):
            if i == 0:
                h = self.conv1(g, in_feat)
                if self.pair_norm:
                    h = PairNorm()(h)
                h = self.activation(h) 
            elif i == self.num_layers - 1:
                h = self.conv3(g, h)
            else:
                h = self.conv2(g, h)
                if self.pair_norm:
                    h = PairNorm()(h)
                h = self.activation(h)  
        return h

class GCNModel_for_LP (nn.Module):
    def __init__(self, in_feats, h_feats, num_layers, activation, dropedge=False, p=1):
        super(GCNModel_for_LP, self).__init__()
        self.num_layers = num_layers
        self.act = activation
        self.DropEdge = dropedge
        self.p = p
        self.conv1 = GCNLayer(in_feats, h_feats)
        self.conv2 = GCNLayer(h_feats, h_feats)
        
    def forward(self, g, in_feat):
        if self.DropEdge:  
            g = dgl.transforms.DropEdge(p=self.p)(g)
        for i in range(self.num_layers):
            if i==0:
                h = self.act(self.conv1(g, in_feat))
                
            elif i != (self.num_layers-1):
                h = self.act(self.conv2(g, h))
                
        return h

