import torch 
from torch import nn
import math


### 最简单的attention版本，只实现对应的公式
class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim: int=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        #(hidden_dim, hidden_dim)
        self.query_mat = nn.Linear(hidden_dim, hidden_dim)
        self.key_mat = nn.Linear(hidden_dim, hidden_dim)
        self.value_mat = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        # x (batch_size, seq_len, hidden_dim)

        # Q K V (batch_size, seq_len, hidden_dim)
        Q = self.query_mat(X)
        K = self.key_mat(X)
        V = self.value_mat(X)
        
        # attention_value Q* KT 
        # KT (batch_size, hidden_dim, seq_len)
        # Q * Kt (batch_size, seq_len, seq_len)
        attention_value = torch.matmul(Q, K.transpose(-1, -2))
        
        # softmax 不影响 矩阵形状， 所以 同 attention_value
        attention_weight = torch.softmax(attention_value / math.sqrt(self.hidden_dim), -1)
        print(attention_weight)
        # output (batch_size, seq_ len, hidden_dim)
        output = torch.matmul(attention_weight, V)
        
        return output 

# X = torch.rand(2, 4,3 )
# model = SelfAttentionV1(3)

# # print(model(X))       

