import torch 
from torch import nn
import math 


### 相比于v1的写法可以优化的点（效率优化）
## 当矩阵比较小的时候，qkv矩阵的运算可以合并成一个大矩阵
class SelfAttentionV2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim =  dim 

        self.qkv_matrix = nn.Linear(dim, dim*3)
    
    def forward(self, X):
        qkv = self.qkv_matrix(X)
        q, k, v = torch.split(qkv, self.dim, dim=-1)

        attention_value = torch.matmul(q, k.transpose(-1,-2))/ math.sqrt(self.dim)

        attention_weight = torch.softmax(attention_value, dim=-1)

        # output = torch.matmul(attetion_weight,v)
        # 等价于下面的@符号
        output = attention_weight @ v

        return output 

X = torch.rand(3,2,4)
model = SelfAttentionV2(4)
print(model(X))