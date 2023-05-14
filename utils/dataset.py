import torch
from torch.utils.data import DataLoader,Dataset

from . import data_helper,sampler

from torch.utils.data.sampler import BatchSampler

import itertools
from functools import partial

class en_to_arr_dataset(Dataset):
    def __init__(self, x_path,y_path,x_vocab=None,y_vocab=None):
        
        print("Reading data from txt files ... ",end=" ")
        with open(x_path) as f:
            self.x_data = f.readlines()
            
        with open(y_path) as f:
            self.y_data = f.readlines()
            
        print("Done!")
            
        # get tokenizers
        self.x_tokenizer,self.y_tokenizer = data_helper._get_tokenizers()
        
        # build vocab 
        x_itr = iter(self.x_data)
        y_itr = iter(self.y_data)
        
        if x_vocab is None and y_vocab is None:
            print("building x_vocab ... ",end = " ")
            self.x_vocab = data_helper._build_vocab(x_itr,self.x_tokenizer)
            print("Done!")
            print("building y_vocab ... ",end = " ")
            self.y_vocab = data_helper._build_vocab(y_itr,self.y_tokenizer)
            print("Done!")
        else:
            print("setting previously calculated vocabs ... ",end=" ")
            self.set_vocabs(x_vocab,y_vocab)
            print("Done!")
        
    def get_vocabs(self):
        return self.x_vocab,self.y_vocab
    
    def set_vocabs(self,x_vocab,y_vocab):
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        
        item_x = self.x_vocab(self.x_tokenizer(self.x_data[idx]))
        item_y = self.y_vocab(["<SOS>"]) + self.y_vocab(self.y_tokenizer(self.y_data[idx])) + self.y_vocab(["<EOS>"])
        
        return (item_x,item_y)
    
    
    
def _get_mask(Y,padding_fill_value):
    mask = []
    
    for seq in Y:
        seq_mask = []
        for token in seq:
            if token == padding_fill_value:
                seq_mask.append(0)
            else:
                seq_mask.append(1)
        mask.append(seq_mask)
        
    return mask
            
def _collate_fn(batch,padding_fill_value):
    # get X and Y
    X = torch.concat( [torch.unsqueeze(torch.tensor(x[0]), 0) for x in batch] ,dim=0 )
    Y = torch.tensor(list(itertools.zip_longest(*[ x[1] for x in batch  ], fillvalue=padding_fill_value))).T
    # get mask
    mask = torch.tensor(_get_mask(Y,padding_fill_value))
    
    return X,Y,mask


def get_data_loader(en_path,ar_path,batch_size,drop_last,x_vocab=None,y_vocab=None):
    
    dataset = en_to_arr_dataset(en_path,ar_path,x_vocab,y_vocab)
    x_vocab,y_vocab = dataset.get_vocabs()
    
    s = sampler.RandomSameLengthSampler(dataset,num_samples=batch_size)
    bs = sampler.CustomBatchSampler(s,batch_size,drop_last)
    
    dl = torch.utils.data.DataLoader(dataset, batch_sampler=bs,collate_fn=partial(_collate_fn,padding_fill_value=y_vocab["<PAD>"]))
    
    return dl,x_vocab,y_vocab