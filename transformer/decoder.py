import torch
from torch import nn

class DecoderBlock(nn.Module):
    def __init__(self, num_head, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_head
        assert hidden_dim == self.head_dim * num_head

        # attention 
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim,hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_atten = nn.Dropout(dropout_rate)
        self.ln_atten = nn.LayerNorm(hidden_dim, eps =0.00001)

        # ffn
        self.up_proj = nn.Linear(hidden_dim, hidden_dim *4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.activate_fn = nn.ReLU()
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.ln_ffn = nn.LayerNorm(hidden_dim,eps = 0.00001)

    def atten_forward(self, x, attention_mask):
        # q (batch_size, seq_len, hidden_dim) => (batch_size, num_heads, seq_len, head_dim)
        _x = x
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_head, -1).transpose(1,2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_head, -1).transpose(1,2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_head, -1).transpose(1,2)

        attention_weight = torch.matmul(
            q, k.transpose(-1,-2)
        ) / torch.sqrt(torch.tensor(self.head_dim))

        if attention_mask is not None:
            attention_mask = attention_mask.tril()
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("-inf")
            )
        else:
            attention_mask = torch.ones_like(attention_weight).tril()
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("-inf")
            )


        attention_weight = torch.softmax(attention_weight, -1)
        attention_weight = self.dropout_atten(attention_weight)

        # (batch_size, num_head, seq_len, head_dim)
        attention_res = attention_weight @ v
        attention_res = attention_res.transpose(1,2).contiguous().view(batch_size, seq_len,-1)

        output = self.output_proj(attention_res)
        return self.ln_atten(output+_x)


    def ffn_forward(self, x):
        _x = x
        x = self.up_proj(x)
        x = self.activate_fn(x)
        x = self.down_proj(x)
        return self.ln_ffn(x + _x)
    def forward(self, x, attention_mask):
        # x (batch_size, seq_len, hidden_dim)
        x = self.atten_forward(x, attention_mask)
        x = self.ffn_forward(x,)
        return x 

x = torch.rand(3,4, 64)
model = DecoderBlock(8,64)
print(model)
mask = (
    torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]])
    .unsqueeze(1)
    .unsqueeze(2)
    .repeat(1, 8, 4, 1)
)
print(mask.shape)
print(model(x, mask).shape)


