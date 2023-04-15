from collections import defaultdict

import torch
from torch.utils.data.sampler import BatchSampler,Sampler

from collections import defaultdict


class RandomSameLengthSampler(Sampler[int]):
    r"""Randomly selects a length category, and samples from it

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source,num_samples):
        self.data_source = data_source
        self.num_samples = num_samples 
        
        # group same length togather
        mapper = defaultdict(list)
        for i,(x,y) in enumerate(self.data_source):
            mapper[len(x)].append(i)
            
        self.original_idxs_categories = mapper.values()
        
        self._reset_iterators()
        
    def _reset_iterators(self):
        self.idxs_categories = [iter(torch.tensor(x)[torch.randperm(len(x))].tolist()) for x in self.original_idxs_categories]
        self.items_left_in_cat = [True]*len(self.idxs_categories)
        
        
    def __iter__(self):
        n = len(self.data_source)
        # randomly select a category from idxs_categories, making sure there are elements left to iterate over
        found = False
        for cat_idx in torch.randperm(len(self.idxs_categories)).tolist():
            found = self.items_left_in_cat[cat_idx]
            if found:
                break
        if not found: # reset iter and select a random category
            self._reset_iterators()
            cat_idx = torch.randint(high=len(self.idxs_categories),size=(1,)).item()
        
        # get elements
        sample_from = self.idxs_categories[cat_idx]
        for _ in range(self.num_samples):
            try:
                yield next(sample_from)
            except StopIteration:
                self.items_left_in_cat[cat_idx] = False
                return # will raise StopIteration exception

    def __len__(self) -> int:
        return len(self.data_source)