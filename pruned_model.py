import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer, PreTrainedTokenizer
from collections import defaultdict
import re
import random

class PruneModel:

    def __init__(self, model, dataset, n=1, tokenizer=None, dropout_threshold=0.5, dropout_seed=123):
        self.dropout_threshold = dropout_threshold
        self.dropout_seed = dropout_seed
        random.seed(dropout_seed)
        self.n = n
        self.model = model
        self.dataset = dataset


        # tokenizer logic
        if tokenizer is None:
            ckpt = getattr(model.config, "name_or_path", None)
            if not ckpt:
                raise ValueError("Please either pass `tokenizer=` or use a pretrained model with config.name_or_path set")
            tokenizer = ckpt

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
            a = acts[l][:, -1, :]
            b = acts[l+self.n][:, -1, :]
            avg_dist.append(self.angular_distance(a, b).mean().item() )
        return int(torch.tensor(avg_dist).argmin().item())

    def prune_state_dict(self):
        #drop whole layer
        pat = re.compile(r"\.layers\.(\d+)\.")
        new_sd = {}
        for k, v in self.model.state_dict().items():
            m = pat.search(k)
            if m:
                idx = int(m.group(1))
                if self.l_star <= idx < self.l_star + self.n:
                    continue
            new_sd[k] = v
        return new_sd
    
    def prune_entire_layer(self, layer: nn.Module):
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(module, name="weight", amount=1.0, n=2, dim=0)
                #prune.ln_structured(module, name="bias",   amount=1.0, n=1, dim=0)

    def angular_distance_prune(self):
        #masks
        layer_len = len(self.model.model.layers)
        upper_threshold = min(layer_len, self.l_star + self.n)
        for i in range(self.l_star, upper_threshold):
            block = self.model.model.layers[i]
            self.prune_entire_layer(block)

        #remove pruning re-parametrization so that state_dict() is updated
        for i in range(self.l_star, upper_threshold):
            block = self.model.model.layers[i]
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    prune.remove(module, "weight")
                    #prune.remove(module, "bias")
    
    def randomized_dropout(self):
        self.dropped_layers = []
        layer_len = len(self.model.model.layers)

        #masks
        for i in range(layer_len):
            curr_dropout_prob = random.uniform(0, 1)
            if (curr_dropout_prob < self.dropout_threshold):
                self.prune_entire_layer(self.model.model.layers[i])
                self.dropped_layers.append(i)
        
        #remove pruning re-parametrization so that state_dict() is updated
        for i in self.dropped_layers:
            block = self.model.model.layers[i]
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    prune.remove(module, "weight")
                    #prune.remove(module, "bias")


    
    