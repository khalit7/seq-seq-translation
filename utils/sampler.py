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
        while any(self.items_left_in_cat):
            # randomly select a category from idxs_categories, making sure there are elements left to iterate over
            found = False
            for cat_idx in torch.randperm(len(self.idxs_categories)).tolist():
                found = self.items_left_in_cat[cat_idx]
                if found:
                    break
            if not found:
                break

            # get elements
            sample_from = self.idxs_categories[cat_idx]
            for _ in range(self.num_samples):
                try:
                    yield next(sample_from)
                except StopIteration:
                    self.items_left_in_cat[cat_idx] = False
                    yield "STOP BATCH ITERATION FOR THIS CATEGORY"
                    break
                    #return # will raise StopIteration exception

    def __len__(self) -> int:
        return len(self.data_source)
    
    
    
    
    
class CustomBatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last):
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) :
        sampler_iter = iter(self.sampler)
        while True:
            try:
                batch = []
                for _ in range(self.batch_size):
                    value = next(sampler_iter)
                    if value == "STOP BATCH ITERATION FOR THIS CATEGORY":
                        break
                    else:
                        batch.append(value)
                if len(batch) == 0: # not allowed to yield an empty batch
                    continue
                if self.drop_last:
                    if len(batch) == self.batch_size:
                        yield batch
                    else:
                        continue
                else: # self.drop_last == False
                    yield batch
            except StopIteration:
                break

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]