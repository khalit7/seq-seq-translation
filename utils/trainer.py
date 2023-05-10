import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2

from torchtext.vocab import vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

from functools import partial

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self,model,train_loader,val_loader,number_of_epochs,criterion,optimizer,scheduler,device,model_path,model_name,writer,starting_epoch=1):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.number_of_epochs = number_of_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion
        self.model_name = model_name
        self.model_path = model_path
        self.writer = writer
        self.starting_epoch=starting_epoch
        
        self.min_val_loss = np.inf
                
        self.num_steps_train = len(self.train_loader)
        self.num_steps_val = len(self.val_loader)
            
            
        if self.starting_epoch == 0 :
            raise Exception(f"cannot start at epoch 0, starting epoch must be 1 or higher")
        if self.starting_epoch > 1: # continuing training ... load latest model
            self.model.load_state_dict(torch.load(model_path))
            
        # send model to device
        self.model.to(self.device)
        
    def train(self):
        
        print(f"TRAINING STARTED using device = {self.device} .... training the model {self.model_name}, the training will continue for {self.number_of_epochs} epochs",end="\n \n")
        for e in range(self.starting_epoch,self.number_of_epochs+1):
            
            print(f"    epoch #{e}")
            
            print(f"        training ...",end=" ")
            epoch_loss = self._train_epoch(e)
            self.writer.add_scalar("train_loss_per_epoch",epoch_loss,e)
            print("DONE")

            print(f"        evaluating ...",end=" ")
            epoch_loss = self._val_epoch(e)
            self.writer.add_scalar("val_loss_per_epoch",epoch_loss,e)
            if epoch_loss < self.min_val_loss:
                print(" Saving best model so far ...",end=" ")
                # save model
                self._save_model()
                # update min loss
                self.min_val_loss = epoch_loss
            
            print("DONE")
            
            if self.scheduler:
                self.scheduler.step()

    def _save_model(self):
        # if self.model_path directory doesn't exist, create directory
        os.makedirs(self.model_path,exist_ok=True)
        # move state dict to cpu
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict,os.path.join(self.model_path,"model.pth"))
        
    def _train_epoch(self,epoch):
        self.model.train()
        
        running_loss = []
        
        for i,(x_source,y,mask) in enumerate(self.train_loader,1):
            # get x_target and y_true
            x_target = y[:,0:-1]
            y_true   = y[:,1:].flatten()
            mask = mask[:,1:].flatten()
            # send tensors to device
            x_source,x_target,y_true,mask = x_source.to(self.device),x_target.to(self.device),y_true.to(self.device),mask.to(self.device)
            # get model's predicitions
            y_pred = self.model(x_source,x_target)
            
            # calculate loss
            loss = self.criterion(y_pred,y_true,mask)
            # reigister running loss
            step_loss = loss.detach().cpu().item()
            self.writer.add_scalar("train_loss_per_step",step_loss,(self.num_steps_train*(epoch-1)) + i)
            running_loss.append(step_loss)
            
            # backward prob
            loss.backward()
            self.optimizer.step()
            
            # empty the gradients
            self.optimizer.zero_grad()
            
            # delete device tensors to free up memory
            del x_source,x_target,y_true,mask,y_pred,loss
        
        epoch_loss = np.mean(running_loss)
        return epoch_loss
    
    def _val_epoch(self,epoch):
        self.model.eval()
        
        running_loss = []
        
        for i,(x_source,y,mask) in enumerate(self.val_loader,1):
            # get x_target and y_true
            x_target = y[:,0:-1]
            y_true   = y[:,1:].flatten()
            mask = mask[:,1:].flatten()
            # send tensors to device
            x_source,x_target,y_true,mask = x_source.to(self.device),x_target.to(self.device),y_true.to(self.device),mask.to(self.device)
            
            with torch.no_grad():
                # get model's predicitions
                y_pred = self.model(x_source,x_target)

                # calculate loss
                loss = self.criterion(y_pred,y_true,mask)

                # reigister running loss
                step_loss = loss.detach().cpu().item()
                self.writer.add_scalar("val_loss_per_step",step_loss,(self.num_steps_val*(epoch-1)) + i)
                running_loss.append(step_loss)
                
            # delete device tensors to free up memory
            del x_source,x_target,y_true,mask,y_pred,loss
        
        epoch_loss = np.mean(running_loss)
        return epoch_loss