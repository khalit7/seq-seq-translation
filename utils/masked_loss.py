import torch
import torch.nn as nn

def masked_crossEntropyLoss(y_pred,y_true,mask):
    loss = nn.CrossEntropyLoss(reduction="none")
    l = loss(y_pred,y_true)
    masked_loss_tensor = torch.masked_select(l,mask>0)
    
    return masked_loss_tensor.mean()