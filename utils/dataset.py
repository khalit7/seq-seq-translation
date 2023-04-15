import torch
from torch.utils.data import DataLoader,Dataset

import data_helper

class en_to_arr_dataset(Dataset):
    def __init__(self, x_path,y_path):
        
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
        
        print("building x_vocab ... ",end = " ")
        self.x_vocab = data_helper._build_vocab(x_itr,self.x_tokenizer)
        print("Done!")
        
        print("building y_vocab ... ",end = " ")
        self.y_vocab = data_helper._build_vocab(y_itr,self.y_tokenizer)
        print("Done!")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        
        item_x = self.x_vocab(self.x_tokenizer(self.x_data[idx]))
        item_y = self.y_vocab(["<SOS>"]) + self.y_vocab(self.y_tokenizer(self.y_data[idx])) + self.y_vocab(["<EOS>"])
        
        return (torch.tensor(item_x),torch.tensor(item_y))