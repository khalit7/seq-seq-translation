import torch


import sys
sys.path.append("utils")
import utils.masked_loss as masked_loss
import utils.dataset as dataset
import utils.models as models
from utils.trainer import Trainer

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter





if __name__ == '__main__':
    model_name = "seq2seq"
    batch_size = 32
    drop_last = False
    embed_size = 300
    hidden_size = 100
    num_layers = 1

    drive_root = Path("drive/MyDrive/seq-seq")
    dataset_root = drive_root / Path("copyable_dataset/train")
    ar_path_train  = dataset_root / "ar-en.ar"
    en_path_train  = dataset_root / "ar-en.en"
    
    dataset_root = drive_root / Path("copyable_dataset/valid")
    ar_path_valid  = dataset_root / "ar-en.ar"
    en_path_valid  = dataset_root / "ar-en.en"

    number_of_epochs = 10
    criterion = masked_loss.masked_crossEntropyLoss
    scheduler = None
    lr = 0.05
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = drive_root / f"weights/{model_name}"


    train_loader,x_vocab, y_vocab = dataset.get_data_loader(en_path_train,ar_path_train,batch_size,drop_last,x_vocab=None,y_vocab=None)
    val_loader,_,_ = dataset.get_data_loader(en_path_valid,ar_path_valid,batch_size,drop_last,x_vocab=x_vocab,y_vocab=y_vocab)

    model = models.seq2seq(len(x_vocab),len(y_vocab),embed_size=embed_size,hidden_size=hidden_size,num_layers=num_layers)
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    
    writer = SummaryWriter(drive_root / f".runs/{model_name}")

    trainer = Trainer(model,train_loader,val_loader,number_of_epochs,criterion,optimizer,scheduler,device,model_path,model_name,writer)
    trainer.train()