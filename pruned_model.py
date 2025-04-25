from data import get_data, DatasetFormat


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
import re

class PruneModel:

    def __init__(self, model, args, datasetformat):
        self.model = model
        self.args = args
        self.datasetformat = datasetformat
        self.dataLoader = self.buildDataLoader()
        self.l_star = self.find_prune_point(self.model, self.dataLoader)


    def collate_fn(batch):
        return {
            "inputs":  [ex.input  for ex in batch],
            "outputs": [ex.output for ex in batch],
        }
    
    def buildDataLoader(self):
        data = get_data(
            random_shuffle=False,
            num_samples=self.args.num_samples,               
            dataset= self.datasetformat, 
            n_shot=0,
            seed=42,
            template=None,
        )
        dl = DataLoader(
            data,
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

    def grab_activations(self, model, dataloader):
        acts = defaultdict(list)
        hooks = []
        for i, blk in enumerate(model.blocks):    
            hooks.append(
                blk.register_forward_pre_hook(
                    lambda mod, inp, idx=i: acts[idx].append(inp[0].detach())
                )
            )
        model.eval()
        with torch.no_grad():
            for xb, *_ in dataloader:
                model(xb)
        for h in hooks: h.remove()
        # stack into tensors of shape [B_total, T, D]
        for i in acts:
            acts[i] = torch.cat(acts[i], dim=0)
        return acts
    
    def find_prune_point(self, model, dataloader, n):
        acts = self.grab_activations(model, dataloader)
        L = len(model.blocks) - n
        avg_dist = []
        for l in range(L):
            a = acts[l][:, -1, :]    # last token
            b = acts[l+n][:, -1, :]
            avg_dist.append(self.angular_distance(a, b).mean().item() )
        return int(torch.tensor(avg_dist).argmin().item())

    def prune_state_dict(self, prefix="blocks", n=1):
        pat = re.compile(rf"{prefix}\.(\d+)\.")
        sd = self.model.state_dict()
        new_sd = {}
        for k, v in sd:
            m = pat.match(k)
            if m and self.l_star <= int(m.group(1)) < self.l_star + n:
                continue
            new_sd[k] = v
        return new_sd