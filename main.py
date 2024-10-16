import torch
# from transformer import LLM
print('welcome chat')
model = torch.load('./scratch_transformer/gpt_500k.pth')
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))