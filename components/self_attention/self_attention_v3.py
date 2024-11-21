import torch 
from torch import nn
import math

### 假如一些细节
## 1. dropout的位置放在哪
## 2. attention mask
## 3. output matrix 


class SelfAttentionV3(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.project_matrix = nn.Linear(self.hidden_dim, self.hidden_dim*3)
        self.output_matrix = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention_dropout = nn.Dropout(dropout_rate)

    def forward(self, X, mask = None):
        qkv = self.project_matrix(X)
        q, k, v = torch.split(qkv, self.hidden_dim, dim = -1)

        # attention weight 
        attention_weight = q @ k.transpose(-1,-2)/ math.sqrt(self.hidden_dim)
        
        # attention mask 
        if mask is not None:
            attetion_weight = attention_weight.masked_fill(
                mask == 0,
                float("-1e20")
            )
        # softmax
        attention_weight = torch.softmax(attention_weight,dim =-1)
        # dropout 
        attention_weight = self.attention_dropout(attetion_weight)
        # attention result
        attention_result = attention_weight @ v
        
        # output matrix 
        output = self.output_matrix(attention_result)
        return output 


X = torch.rand(3,4,2)
attention_mask = torch.tensor(
    [
        [1,1,1,0],
        [1,1,0,0],
        [1,0,0,0],
    ]
)
print(f"old attention mask size is {attention_mask.size()}")
attention_mask = attention_mask.unsqueeze(dim=1).repeat(1,4,1)
print(f"mask shape is {attention_mask.size()}")
model = SelfAttentionV3(2)
print(model(X,attention_mask))