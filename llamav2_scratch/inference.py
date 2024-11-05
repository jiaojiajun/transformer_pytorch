import torch
import time
from typing import Optional, List 
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import Transformer, ModelArgs

class LLaMa:
    def __init__(self, model: Transformer, tokenizer:SentencePieceProcessor, model_args:ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
    
    @staticmethod
    def build(checkpoints_dir:str, tokenizer_path:str, load_model:bool, max_seq_len:int, max_batch_size:int, device:str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkponit files found in path {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'load checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f'loaded checkpoint in {time.time()-prev_time:.2f}s')
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", 'r' ) as f:
            params = json.loads(f.read())
        
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"loaded state dict in {  time.time() - prev_time:.2f}s")
        return LLaMa(model, tokenizer, model_args)
    
    def text_completion(self, prompts:List[str], device:str, temperature:float=0.6, top_p=0.9, max_gen_len:Optional[int]=None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len -1
        prompts_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts ]
        batch_size = len(prompts)
        assert batch_size < self.args.max_batch_size, "too many prompts to be processed at a time"
        max_prompt_len = max(len(tokens) for tokens in prompts_tokens)
        assert max_prompt_len < self.args.max_seq_len, f"prompt token size must <= {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len+max_prompt_len)

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id,dtype=torch.long, device=device)
        for idx, prompt_tokens in enumerate(prompts_tokens):
            tokens[idx,:len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False]* batch_size, device=device)
        prompt_tokens_mask = tokens!=pad_id
        cur_iter = tqdm(range(1, total_len), desc="generating tokens")
        for cur_pos in cur_iter:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1: cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1]/temperature, dim=-1)
                next_token = self._sample_top_p(probs,top_p)
            else:
                next_token = torch.argmax(logits[:,-1], dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_tokens_mask[:,cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) &(next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_index = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_index]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text
    
    def _sample_top_p(self, probs:torch.tensor, top_p:float):
        sorted_probs,probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(sorted_probs, dim=-1)
        mask = probs_sum - sorted_probs > top_p
        sorted_probs[mask] = 0.0 
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

if __name__ == "__main__":
    torch.manual_seed(0)
    
    allow_cuda = True  
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"


    prompts = [
        "Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from ",
        # "the capital of usa is ",
        # "the rpc is ",
        # # Few shot promt
        # """Translate English to French:
        
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
        # # Zero shot prompt
        # """Tell me if the following person is actually Doraemon disguised as human:
        # Name: march json
        # Decision: """
    ]
    model = LLaMa.build('Llama-2-7b/',
                        tokenizer_path='Llama-2-7b/tokenizer.model',
                        load_model=True,
                        max_seq_len=1024,
                        max_batch_size=2,
                        device=device
                        )
    out_tokens, out_text  = model.text_completion(prompts,max_gen_len=100, device=device)
    assert len(out_text) == len(prompts)
    print(out_text)