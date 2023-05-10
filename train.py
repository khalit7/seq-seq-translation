import torch
import yaml

from utils import helper,dataset,trainer

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter





if __name__ == '__main__':
    
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    # get hyperparameters
    model_name = config["model_name"]
    batch_size = config["batch_size"]
    drop_last = config["drop_last"]
    embed_size = config["embed_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    number_of_epochs = config["number_of_epochs"]
    scheduler_str = config["scheduler"]
    lr = config["lr"]
    is_continue_training = config["is_continue_training"]
                       
    model_path = f"weights/{model_name}"
    checkpoint_path = f"checkpoint/{model_name}"
    summary_writer_path = f".runs/{model_name}"
    
    train_dataset_root = Path(config["dataset_root"])/"train"
    ar_path_train  = train_dataset_root / "ar-en.ar"
    en_path_train  = train_dataset_root / "ar-en.en"
    
    valid_dataset_root = Path(config["dataset_root"])/"valid"
    ar_path_valid  = valid_dataset_root / "ar-en.ar"
    en_path_valid  = valid_dataset_root / "ar-en.en"
    
    # setup device
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.has_mps:
        device_str = "mps"
    else:
        device_str = "cpu"
    device_str = "cpu"
    device = torch.device(device_str)
    
    # get train and valid data                  
    train_loader,x_vocab, y_vocab = dataset.get_data_loader(en_path_train,ar_path_train,batch_size,drop_last,x_vocab=None,y_vocab=None)
    val_loader,_,_ = dataset.get_data_loader(en_path_valid,ar_path_valid,batch_size,drop_last,x_vocab=x_vocab,y_vocab=y_vocab)

    # get model 
    model_kwargs ={
    "x_vocab_size":len(x_vocab),
    "y_vocab_size":len(y_vocab),
    "embed_size":embed_size,
    "hidden_size":hidden_size,
    "num_layers":num_layers
    }
    model = helper.get_model_by_name(model_name,**model_kwargs)
                       
    # get optimizer
    optimizer = helper.get_optimizer(model,lr)
    # get scheduler
    scheduler = helper.get_scheduler_by_name(scheduler_str,optimizer)
    # loss criterion
    criterion = helper.get_criterion()
    # init tensorbaord summary writer
    writer = SummaryWriter(summary_writer_path)

    # get the trainer and train
    trainer = trainer.Trainer(model,train_loader,val_loader,number_of_epochs,criterion,optimizer,scheduler,device,model_path,checkpoint_path,model_name,writer,continue_training=is_continue_training)
    trainer.train()