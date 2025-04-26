import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizer
from collections import defaultdict
import re

class PruneModel:

    def __init__(self, model, dataset, n=1, tokenizer=None):
        self.n = n
        self.model = model
        self.dataset = dataset


        # tokenizer logic
        if tokenizer is None:
            ckpt = getattr(model.config, "name_or_path", None)
            if not ckpt:
                raise ValueError("Please either pass `tokenizer=` or use a pretrained model with config.name_or_path set")
            tokenizer = ckpt

        # allow either a tokenizer instance or a string name
        if isinstance(tokenizer, PreTrainedTokenizer):
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        if self.tokenizer.pad_token is None:
            # use eos_token as pad, if available
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # else: create a new [PAD] token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.dataloader = self.buildDataLoader()
        self.l_star = self.find_prune_point()


    def collate_fn(self, batch):
        texts = [ex.input for ex in batch]
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
    
    def buildDataLoader(self):
        dl = DataLoader(
            self.dataset,
            batch_size=8, 
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        return dl
    
    def angular_distance(self, a, b):
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        cos = (a_n * b_n).sum(-1).clamp(-1,+1)
        return (1.0 / torch.pi) * torch.acos(cos)  

    def grab_activations(self):
        acts  = defaultdict(list)
        hooks = []
        for i, blk in enumerate(self.model.model.layers):
            hooks.append(
                blk.register_forward_pre_hook(
                    lambda mod, inp, idx=i: acts[idx].append(inp[0].detach().cpu())
                )
            )

        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                batch = {k: t.to(device) for k, t in batch.items()}
                self.model(**batch)

        for h in hooks: h.remove()
        for i in acts:
            acts[i] = torch.cat(acts[i], dim=0)
        return acts
    
    def find_prune_point(self):
        acts = self.grab_activations()
        L = len(self.model.model.layers) - self.n
        avg_dist = []
        for l in range(L):
            a = acts[l][:, -1, :]    # last token
            b = acts[l+self.n][:, -1, :]
            avg_dist.append(self.angular_distance(a, b).mean().item() )
        return int(torch.tensor(avg_dist).argmin().item())

    def prune_state_dict(self, prefix="self_attn"):
        pat = re.compile(rf"{prefix}\.(\d+)\.")
        sd = self.model.state_dict()
        new_sd = {}
        for k, v in sd.items():
            m = pat.match(k)
            if m and self.l_star <= int(m.group(1)) < self.l_star + self.n:
                continue
            new_sd[k] = v
        return new_sd