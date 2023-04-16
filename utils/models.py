import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(self.vocab_size,self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
        
        
    def forward(self,x):
        x                = self.embed(x)
        output,(h_n,c_n) =  self.lstm(x)
        
        return output,(h_n,c_n)
    
    
class Decoder(nn.Module):
    
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(self.vocab_size,self.embed_size)
        self.lstm  = nn.LSTM(input_size=self.embed_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
        self.fc    = nn.Linear(self.hidden_size,self.vocab_size)
        
        
    def forward(self,x,h_n,c_n):
        x = self.embed(x)
        output, (hn, cn) = self.lstm(x,(h_n,c_n))
        logits = self.fc(output)
        
        return logits.reshape(-1,self.vocab_size)
    
class seq2seq(nn.Module):
    
    def __init__(self,x_vocab_size,y_vocab_size,embed_size,hidden_size,num_layers):
        super().__init__()
        self.encoder = Encoder(x_vocab_size,embed_size,hidden_size,num_layers)
        self.decoder = Decoder(y_vocab_size,embed_size,hidden_size,num_layers)
        
    def forward(self,x_source,x_target):
        output,(h_n,c_n) = self.encoder(x_source)
        logits           = self.decoder(x_target,h_n,c_n)
        
        return logits
        
    