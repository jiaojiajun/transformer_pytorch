from dataclasses import dataclass 
from typing import Optional 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

@dataclass
class ModelArgs:
    dim:int = 4096
    n_heads: int = 32
    n_layers: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # it should be set int build method 
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float]= None
    norm_eps: float = 1e-5

    ## kv cache args 
    max_batch_size:int = 32
    max_seq_len:int =2048
    

    device: str = None 

class RMSNorm(nn.Module):
    def __init__(self, dim, eps= 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def _rms(self,x:torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+ self.eps)

    def forward(self, x:torch.Tensor):
        return self.weight * self._rms(x.float()).type_as(x)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, " vocab_size must be set"


        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)#todo parameter to add

    def forward(self, tokens:torch.Tensor, start_pos:int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token can be processed at the same time"

        h = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[start_pos: start_pos+seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output

# 到这里我们还需要实现什么？
# 1. EncoderBlock
# 2. precompute_theta_pos_frequencies

## 先来实现 encoder block， 参考llama模型图，我们需要实现的是
### 1.  self attention 
### 2. apply rotary position encoding 
### 3. res net 
### 4. ffn 
### 让我们一步步来实现这个

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # multihead
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // self.n_heads
        


        self.attention_norm = RMSNorm(args.dim)
        self.attention = SelfAttention(args)

        self.ffn_norm = RMSNorm(args.dim)
        self.feed_forward = FeedForward(args)

    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex:torch.Tensor):
        ## x shape is (batch_size, seq_len, dim)
        ## (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        _x = x 
        x = self.attention_norm(x)
        x = self.attention(x,start_pos, freqs_complex)
        x = _x + x 
        _x = x 
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        return _x + x 
        ## 上面的写法可以方便我们看清楚每一个步骤，实际上可以写的简略些
        # h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)

        # return h + self.feed_forward(self.ffn_norm(x))

## 梳理一下未实现的东西
## 1. self attention
## 2. apply position_encoding
## 4. feed_forward

## 我们这里的注意力机制采用的是与llama相同的分组注意力机制。同时为了实现简单，我们仅仅把kv 复制
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        ## multi head and group query 
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        # how many times should kv repeat 
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // self.n_heads

        ## project matrix 
        self.wq = nn.Linear(args.dim, self.head_dim * self.n_heads, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wo = nn.Linear(self.head_dim * self.n_heads, args.dim, bias=False)

        ## kv cache
        self.cache_k = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads, self.head_dim))
    
    def forward(self,x:torch.Tensor, start_pos:int, freqs_complex: torch.tensor):
        batch_size, seq_len, _ = x.shape
        
        # project the matrix 
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # view q k v
        # (batch_size, seq_len, dim) ->(batch_size, seq_len, n_heads, head_dim)
        xq = xq.view(batch_size, seq_len, -1, self.head_dim)
        # (batch_size, seq_len, dim) ->(batch_size, seq_len, n_kv_heads, head_dim)
        xk = xk.view(batch_size, seq_len, -1, self.head_dim)
        xv = xv.view(batch_size, seq_len, -1, self.head_dim)

        ## apply rotary position encoding 
        # in this step, size won't change 
        xq = apply_rotary_encoding(xq, freqs_complex,device=x.device)
        xk = apply_rotary_encoding(xk, freqs_complex, device=x.device)

        ## save current k, v in cache
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        ## load past and current keys and values
        keys = self.cache_k[:batch_size, :start_pos+seq_len]
        values = self.cache_v[:batch_size, :start_pos+seq_len]

        ## just simply repeat the keys and values
        keys = repeat_kv(keys,self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # transpose
        # (batch_size, seq_len, n_heads, head_dim)->(batch_size,n_heads,seq_len,head_dim)
        xq = xq.transpose(1,2)
        # (batch_size, start_pos+seq_len, n_kv_heads, head_dim) -> (batch_size, n_kv_heads, start_pos+seq_len, head_dim)
        keys = keys.transpose(1,2).to(x.device)
        values = values.transpose(1,2).to(x.device)

        attention_score = xq @ keys.transpose(-2,-1) / math.sqrt(self.head_dim)
        attention_score = F.softmax(attention_score.float(), dim=-1).type_as(xq)
        output = attention_score @ values

        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


# 梳理一下没有实现的东西
# 1. apply_rotary_encoding
# 2. ffn

# 我们先把ffn实现了
class FeedForward(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        xV = self.w3(x)
        x = swish * xV
        return self.w2(x)

# 梳理一下现在还剩什么没有实现
# 1. precompute_theta_pos_frequencies
# 2. apply_rotary_encoding
# 3. repeat_kv 

def precompute_theta_pos_frequencies(head_dim:int, seq_len:int, device:str, theta:float = 10000.0 ):
    assert head_dim % 2 == 0, "head dim must be even"
    theta_iter = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_iter / head_dim)).to(device)
    m = torch.arange(0, seq_len,device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_encoding(x:torch.Tensor, freqs_complex:torch.Tensor, device:str):

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x:torch.Tensor, n_rep:int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x.unsqueeze(3).expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)

## ok 到这里我们就基本实现完成了
