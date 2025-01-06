
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input

class GLU(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, (1, 1))
        self.conv2 = nn.Conv2d(dim, dim, (1, 1))
        self.conv3 = nn.Conv2d(dim, dim, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out

class NFI(nn.Module):
    """Node stochastic interaction"""
    def __init__(self, d_c, d_f):
        super(NFI, self).__init__()

        self.fc1 = nn.Linear(d_c, d_c)
        self.fc2 = nn.Linear(d_c, d_f)
        self.fc3 = nn.Linear(d_c + d_f, d_c)
        self.fc4 = nn.Linear(d_c, d_c)

    def forward(self, input: torch.Tensor):
        b, n, c = input.shape
        com_feat = F.gelu(self.fc1(input))
        com_feat = self.fc2(com_feat)

        if self.training:
            # b,n,d
            ratio = F.softmax(com_feat, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, n)#b*d,n
            indices = torch.multinomial(ratio, 1)#b*d,1
            indices = indices.view(b, -1, 1).permute(0, 2, 1)#b,1,d
            com_feat = torch.gather(com_feat, 1, indices)#b,1,d
            com_feat = com_feat.repeat(1, n, 1)#b,n,d
        else:
            weight = F.softmax(com_feat, dim=1)
            com_feat = torch.sum(com_feat * weight, dim=1, keepdim=True).repeat(1, n, 1)

        com_feat_new = torch.cat([input, com_feat], -1)
        com_feat_new = F.gelu(self.fc3(com_feat_new))
        output = self.fc4(com_feat_new)

        return output

class Conv(nn.Module):
    def __init__(self, in_dim,out_dim, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=(1,1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class Attention(nn.Module):
    def __init__(self,num_nodes,num_heads,dim,pool_size,attn_drop=0.1,seq_len=1):
        super(Attention,self).__init__()
        self.num_heads = num_heads
        self.qkv = Conv(dim, dim * 3)
        self.pool_size = pool_size
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.nfi = NFI(dim, dim)
        self.weight = nn.Parameter(torch.ones(dim,num_nodes, seq_len))
        self.bias = nn.Parameter(torch.zeros(dim,num_nodes, seq_len))

    def forward(self, x: torch.Tensor):
        # x:B,C,N,T
        b, c, n, t = x.shape
        head_dim = c // self.num_heads
        scale = head_dim ** -0.5
        qkv = self.qkv(x).reshape(b, c, 3, n, t).permute(2,0,4,3,1)
        query, key, value = qkv[0], qkv[1], qkv[2] #b,1,n,c
        pool = self.pool(query)#b,1,p,c

        query = query.reshape(b*self.num_heads, t, n, head_dim)
        key = key.reshape(b*self.num_heads, t, n, head_dim)
        value = value.reshape(b*self.num_heads, t, n, head_dim)
        pool = pool.reshape(b*self.num_heads,t, self.pool_size, head_dim)

        # aggregate
        pool_att = F.softmax((pool * scale) @ key.transpose(-2, -1),dim=-1)# b*num_heads,t,p,n
        pool_att = self.attn_drop(pool_att)#...,p,n
        pool_val = pool_att @ value#b*num_heads,t,p,head_dim

        # broadcast
        query_att = F.softmax((query * scale) @ pool.transpose(-2, -1),dim=-1)
        query_att = self.attn_drop(query_att)# b*num_heads,t,n,p
        att_out = query_att @ pool_val

        value = self.nfi(value.reshape(b,n,c)).reshape(b,t,n,c)
        att_out = att_out.reshape(b,t,n,c)
        att_out = att_out + value
        att_out = rearrange(att_out,'B T N C -> B C N T')

        weight = self.weight.to(x.device)
        bias = self.bias.to(x.device)
        if n in [307,716,883,3834,2352]:
            att_out = att_out * weight + bias + att_out
        return att_out, weight, bias
    
class TemporalEmbedding(nn.Module):
    def __init__(self, time, dim):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, dim))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, dim))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]  
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]  
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]  
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]  
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb
    

class Att_Encoder(nn.Module):
    def __init__(self,num_nodes,num_heads,dim,pool_size,attn_drop=0.1,seq_len=1,proj_drop=0.1):
        super(Att_Encoder, self).__init__()
        self.num_nodes = num_nodes
        self.seq_length = seq_len
        self.d_model = dim
        self.attention = Attention(num_nodes,num_heads,dim,pool_size,attn_drop,seq_len)
        self.ln1 = LayerNorm([dim,num_nodes,seq_len],elementwise_affine=False)
        self.dropout1 = nn.Dropout(p=proj_drop)
        self.glu = GLU(dim)
        self.dropout2 = nn.Dropout(p=proj_drop)

    def forward(self, input):
        x, weight, bias = self.attention(input)
        x = x + input
        x = self.ln1(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * weight + bias + x
        x = self.ln1(x)
        x = self.dropout2(x)
        return x
    
class Model(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            in_dim: int = 3,
            model_dim: int = 128,
            time: int = 288,
            in_len: int = 12,
            out_len: int = 12,
            num_heads: int = 4,
            pool_size: float = 64,
            attn_drop: float = 0.1,
            proj_drop: float = 0.15,
    ):
        super(Model, self).__init__()
        
        if pool_size > 1:
            pool_size = int(pool_size)
        else:
            raise ValueError("pool_size must be a positive number")

        self.start = nn.Linear(1,model_dim)

        self.te = TemporalEmbedding(time, model_dim)

        # time conv
        self.agg_conv = nn.Conv2d(model_dim, model_dim, kernel_size=(1,in_len))

        net_dim = model_dim*2
        # SAE
        self.att_encoder = Att_Encoder(num_nodes,num_heads,net_dim,pool_size,attn_drop,1,proj_drop)

        self.conv = nn.Conv2d(net_dim,net_dim,kernel_size=(1,1))

        self.decoder = nn.Conv2d(net_dim,out_len,kernel_size=(1,1))

    def forward(self, x: torch.Tensor):
        # x:(B,T,N,C)
        time_x = x.detach().clone()

        x_hidd = self.start(x[...,0:1]) #(B,T,N,C)

        x_hidd = rearrange(x_hidd,'B T N C -> B C N T')

        x_hidd = self.agg_conv(x_hidd)#(B,C,N,1)

        # Embedding
        te = self.te(time_x)#(B,C,N,1)

        feat_cat = torch.cat([x_hidd,te],dim=1)

        feat = self.att_encoder(feat_cat) + self.conv(feat_cat)

        out = self.decoder(feat)#(B,C,N,1)

        return out
    
if __name__ == '__main__':
    # model = EnEmbedding(170,128,4,0.1)
    x = torch.randn(16,12,170,3)
    # out = model(x)
    # print(out.shape)
    