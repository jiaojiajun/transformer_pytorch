# in this file we will create simple gpt(decoder only) from scratch

## accordings




import torch 
from torch import nn
from torch.nn import functional as F 


# load data
with open('meta_goer.txt','r',encoding='gb2312', errors='ignore') as f:
    text= f.read()
# print(text[:1000])
# print(len(text))

# raise Exception("pause")

## process 
text = [ch for ch in text if ch != ' ' and ch != '\n']

### make vocab_table 
tokens = sorted(list(set(text)))
stoi = {token:i for i,token in enumerate(tokens)}
itos = {i:token for i,token in enumerate(tokens)}
def encode(str):
    return [stoi[ch] for ch in str ]
def decode(ids):
    return [itos[i] for i in ids]

### encode 
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]



# hyper parameter
head_num = 8
hidden_dim = 64
vocab_size = len(tokens)
seq_len = 256
batch_size = 8
decoder_num = 6
eval_iters = 200
learning_rate = 1e-4
epochs = 5000
eval_internal = 100

# print(vocab_size)

def get_batch(data_type):
    data = train_data if data_type == 'train' else val_data 
    rand_start = torch.randint(len(data)- seq_len,(batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in rand_start] )
    y = torch.stack([data[i+1:i+seq_len+1] for i in rand_start])
    return x, y 

## test get batch method
x_1, y_1 = get_batch('train')
# print(x_1.shape)

## get mean loss of 200 token eval
@torch.no_grad()
def get_mean_loss():
    loss_out = {}
    model.eval()
    for data_type in ['train','validate']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(data_type)
            logis, loss = model(X,Y)
            losses[i] = loss.item()
        loss_out[data_type] = losses.mean()
    model.train()
    return loss_out



class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // head_num

        assert hidden_dim == head_num * self.head_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim) 
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_atten = nn.Dropout(dropout_rate)
        self.dropout_fn = nn.Dropout(dropout_rate)

    def forward(self, x, mask = None):
        batch_size, seq_len, _ = x.size()
        # q k v (batch_size, head_num, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)

        # q @ k.T
        # attention_weight (batch_size, head_num, seq_len, seq_len)
        attention_weight = q @ k.transpose(-2,-1) /torch.sqrt(torch.tensor(self.head_dim))
        # mask
        if mask is not None:
            mask = mask.tril()
            attention_weight = attention_weight.masked_fill(
                mask == 0,float("-inf")
            )
        else:
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1).repeat(batch_size,self.head_num,1,1)
            # print(mask.shape)
            # print(attention_weight.shape)
            attention_weight = attention_weight.masked_fill(
                mask == 0, float("-inf")
            )
        # softmax 
        attention_weight = torch.softmax(attention_weight, -1)
        # drop_out
        attention_weight = self.dropout_atten(attention_weight)
        
        # apply attention weight, dim of attention_value (batch_size, head_num, seq_len, head_dim)
        attention_result = attention_weight @ v

        # concat and output project 
        output = self.output_proj(attention_result.transpose(1,2).contiguous().view(batch_size, seq_len,-1))
        output = self.dropout_fn(output)
        return output

# ## test for multiheadAttention
# x = torch.rand(4,8,32)
# model = MultiHeadAttention(8,32)
# print(model)
# print(model(x))

class FFN(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act_fn = nn.ReLU()
        self.down_proj = nn.Linear(4 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention(head_num, hidden_dim)
        self.ffn = FFN(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x 

class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(seq_len,hidden_dim)
        self.decoders = nn.Sequential(
            *[DecoderLayer() for i in range(decoder_num)]
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs,target=None):
        # inputs size (batch_size, seq_len)
        batch_size, seq_len = inputs.size()
        vocab = self.vocab_embedding(inputs)
        position = self.position_embedding(torch.arange(seq_len))
        embedding = vocab + position 
        x = self.decoders(embedding)
        x = self.ln(x)
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)
        
        if target is  None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.size()

            logits = logits.view(batch_size*seq_len, -1)
            target = target.view(-1)
            loss = F.cross_entropy(logits, target)
        return logits, loss 

    def generate(self,inputs,max_new_tokens):
        # inputs (batch_size, seq_len)
        for _ in range(max_new_tokens):
            _inputs = inputs[:,-seq_len:] # 每个批次只取seq_len个token
            logits, loss = self.forward(_inputs)
            # get the last token 
            logits = logits[:,-1,:]
            # softmaxt
            probs = torch.softmax(logits, -1)
            # get new token
            output_token = torch.multinomial(probs, num_samples = 1) # batch_size,1
            # concat to inputs
            inputs = torch.cat((inputs, output_token), dim=1)
        return inputs

model = LLM()
# print(model)

# for name, param in model.parameters():
#     print(f"name is {name}, and requires grad {param.requires_grad}")
# aaaa
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() )

# print(count_parameters(model)) # 593557

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# train
for epoch in range(epochs):
    if epoch % eval_internal == 0 or epoch == epochs - 1:
        loss = get_mean_loss()
        print(f"epoch: {epoch}, train loss: {loss['train']:.4f}, val loss: {loss['validate']:.4f}")
        
        if epoch >=4000:
            torch.save(model,f"./models/gpt_500k-{epoch}.pth")
    
    x_batch, y_batch = get_batch('train')
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model,'gpt_500k.pth')

context = torch.zeros((1, 1), dtype=torch.long)
print(''.join(decode(model.generate(context, max_new_tokens=2000)[0].tolist())))