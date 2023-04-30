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
    
    
    
    
    
class Node():
    def __init__(self,token,prob,h_n,c_n,prev):
        self.token = token
        self.prob = prob
        self.h_n = h_n
        self.c_n = c_n
        self.prev = prev
        
        if self.prev is not None:
            self.acc_prob = self.prob + self.prev.acc_prob
        else:
            self.acc_prob = self.prob
            
        self.acc_prob_per_token = self.calculate_acc_prob_per_token()
            
    def calculate_acc_prob_per_token(self):
        counter = 1
        prev_node = self.prev
        while prev_node is not None:
            counter+=1
            prev_node = prev_node.prev
            
        return self.acc_prob/counter
    
    
class InferenceSeq2seq(seq2seq):
    
    def __init__(self,x_vocab_size,y_vocab_size,embed_size,hidden_size,num_layers,y_vocab,k=5):
        super().__init__(x_vocab_size,y_vocab_size,embed_size,hidden_size,num_layers)
        
        self.k = k
        self.y_vocab = y_vocab
        
    def forward(self,x_source):
        # encode the input sentence
        output,(h_n,c_n) = self.encoder(x_source)
        
        #
        sos_int = torch.tensor(self.y_vocab(["<SOS>"])).reshape(1,-1)
        sos_node = Node(token=sos_int,prob=0,h_n=h_n,c_n=c_n,prev=None)
        
        top_k_nodes = self.get_top_k(sos_node)   
        
        candidate_nodes = []
        while len(candidate_nodes) < self.k:
            # initialize beam search variables
            k_times_k_nodes = []
            for i,node in enumerate(top_k_nodes):
                if self.y_vocab.lookup_token(node.token.flatten().item()) == "<EOS>":
                    candidate_nodes.append(node)
                    continue
                curr_top_k_nodes = self.get_top_k(node)
                k_times_k_nodes.extend(curr_top_k_nodes)

            # aggregate results so that u have best k nodes so far
            top_k_nodes = self.beam_step(k_times_k_nodes)
        
        return sorted(candidate_nodes,key = lambda x:x.acc_prob_per_token,reverse=True)
        
    
    def get_top_k(self,node):
        # 
        token,h_n,c_n = node.token,node.h_n,node.c_n
        
        x = self.decoder.embed(token)
        output, (h_n, c_n)  = self.decoder.lstm(x,(h_n,c_n))
        logits = self.decoder.fc(output).flatten()
        probs = torch.softmax(logits,dim=0)
        ## pick top k tokens
        top_k = torch.topk(probs,k=self.k)
        top_k_tokens = top_k.indices
        top_k_probs = torch.log(top_k.values)
        
        top_k_nodes = []
        for i in range(self.k):
            curr_node =  Node(token=top_k_tokens[i].reshape(1,-1),prob=top_k_probs[i],h_n=h_n,c_n=c_n,prev=node)
            top_k_nodes.append(curr_node)
        return top_k_nodes
    
    def beam_step(self,k_times_k_nodes):
        return sorted(k_times_k_nodes,key = lambda x:x.acc_prob_per_token,reverse=True)[0:self.k]
        
    