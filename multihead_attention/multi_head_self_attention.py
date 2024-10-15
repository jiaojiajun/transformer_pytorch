import torch 
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, hidden_dim, dropout_rate = 0.1):
        super().__init__()
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // head_num
        assert hidden_dim == self.head_dim * head_num

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask = None):
        # size of x (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.size()

        # initial q, k, v
        # (batch_size, seq_len, hidden_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # divide into multihead
        ## (batch_size, seq_len, hidden_dim) => (batch_size, seq_len, head_num, head_dim) => (batch_size, head_num, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)

        # compute attention 
        # (batch_size, head_num, seq_len, seq_len)
        attention_weight = torch.matmul(
            q, k.transpose(-1,-2)
        )/ torch.sqrt(torch.tensor(self.head_dim))

        # apply mask 
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("-inf")
            )

        # softmax and dropout
        attention_weight = torch.softmax(attention_weight, -1)
        attention_weight = self.dropout(attention_weight)

        # attention apply
        # (batch_size, head_num, seq_len, head_dim)
        attention_result = attention_weight @ v  

        # transpose 
        attention_result = attention_result.transpose(1,2).contiguous().view(batch_size, seq_len, -1)

        # output project
        # (batch_size, seq_len, hidden_dim)
        output = self.output_proj(attention_result)
        return output



x = torch.rand(3,2,64)
model = MultiHeadAttention(8, 64)
print(model(x))
print(model)
