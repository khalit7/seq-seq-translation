from . import models,masked_loss
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
    
def get_model_by_name(model_name,**kwargs):
    if model_name == "seq2seq_rnn":
        return models.seq2seq_rnn(**kwargs)
    else:
        raise Exception(f"the model name you chose was {model_name}, error ... unrecognized model name")
        
def get_criterion():
    return masked_loss.masked_crossEntropyLoss

def get_optimizer(model,lr):
    return torch.optim.SGD(model.parameters(),lr=lr)

def get_scheduler_by_name(name,optimizer):
    if name == "StepLR":
        return StepLR(optimizer, step_size=2, gamma=0.75)
    else:
        raise Exception(f"the scheduler name you chose was {model_name}, error ... unrecognized scheduler name")
    