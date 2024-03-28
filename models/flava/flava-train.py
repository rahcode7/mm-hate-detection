import logging
import time
import torch
from torchvision import transforms
from collections import defaultdict
from transformers import BertTokenizer
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic 
import argparse
import wandb
import os 
from torch.utils.data import DataLoader
from accelerate import Accelerator
import bitsandbytes as bnb
from torchmultimodal.models.flava.model import flava_model_for_classification
from MM_data_loader import FBHMDataset,collate_fn
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm 
import json 
import shutil
import site
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, FlavaMultimodalModel, AutoProcessor,FlavaModel
import torch

#from utils.helpers import set_seed


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    ic("prev checkpoint loaded",checkpoint['epoch'])
    return model, optimizer, checkpoint['epoch']


class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)

def train(epoch,model,train_dataloader,criterion,tokenizer,processor,optimizer,scheduler,device,accelerator,MACHINE_TYPE,ACCUMULATION_STEPS,LR_FLAG):
    model.train()

    train_losses = []
    train_corrects = []
    train_size = [] 
    c = 0 
    classification_head = ClassificationHead(768,num_classes=2)
    classification_head = classification_head.to(device)


    for idx, (files,images,labels) in enumerate(tqdm(train_dataloader)):
        c+=1
        if c>5:
            break
        ic(images,labels)
        with accelerator.accumulate(model):
            text = ["",""]#"",""]
            inputs = processor(text = text, images=images,return_tensors="pt",padding=True)
            #ic(inputs)
            inputs = inputs.to(device)
            labels = torch.FloatTensor(labels)
            labels = labels.to(device)
            outputs = model(**inputs)
            #ic(outputs.multimodal_output.pooler_output)
            ic(outputs.multimodal_output.pooler_output.shape)
            
            classifier_inputs = outputs.multimodal_output.pooler_output.to(device)

            logits = classification_head(classifier_inputs)

            predicted = logits.argmax(dim=1).float()
            ic(predicted,labels)
            
            loss = criterion(predicted,labels)
            ic(loss)

            ## DDP code
            train_losses.append(accelerator.gather(loss))
            #ic(outputs.last_hidden_state)

            # preds = model.module.generate(**inputs,max_new_tokens=100)
            # predicted = processor.batch_decode(preds, skip_special_tokens=True)
            train_batch_corrects = len([i for i,
                                            j in zip(predicted, labels) if i == j])
            ic(train_batch_corrects)
            train_batch_corrects = torch.tensor(train_batch_corrects).to(device)
            # #ic(predicted,outputs.loss,train_batch_corrects)
            train_corrects.append(accelerator.gather(train_batch_corrects))

            label_size = torch.tensor(len(labels)).to(device)
            gathered_sizes = accelerator.gather(label_size)
            train_size.append(gathered_sizes)

            #ic(answers,len(labels),predicted,outputs.loss,gathered_sizes,train_size)
            loss.requires_grad = True
            accelerator.backward(loss)
            # Gradient accumulation 
            #if (idx + 1)% ACCUMULATION_STEPS == 0:
            optimizer.step()
            
            #scheduler.step()
            optimizer.zero_grad()


    # Call every epoch
    if LR_FLAG:
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: AdamW lr %.10f -> %.10f" % (epoch, before_lr, after_lr))

    total_train_size = torch.sum(torch.cat(train_size)).item()
    total_corrects = torch.sum(torch.cat(train_corrects)).item()

    ic(device,total_corrects,total_train_size)
    train_loss = torch.sum(torch.cat(train_losses)).item() / len(train_dataloader)

    #train_accuracy = torch.sum(torch.cat(train_corrects)) / len(train_dataloader.dataset)
    train_accuracy =  total_corrects / total_train_size

    # ic(train_loss,train_accuracy, torch.sum(torch.cat(train_corrects)))
    return train_loss,train_accuracy
    #return None,None 

def evaluate(epoch,model,val_dataloader,criterion,tokenizer,processor,device,accelerator,MACHINE_TYPE,ACCUMULATION_STEPS,LR_FLAG):
    model.train()

    val_losses = []
    val_corrects = []
    val_size = [] 
    c = 0 
    classification_head = ClassificationHead(768,num_classes=2)
    classification_head = classification_head.to(device)
    
    with torch.no_grad():
        model.eval()
        for idx, (files,images,labels) in enumerate(tqdm(val_dataloader)):
            c+=1
            if c>5:
                break
            
            text = ["",""]
            inputs = processor(text = text, images=images,return_tensors="pt",padding=True)
            #ic(inputs)
            inputs = inputs.to(device)
            labels = torch.FloatTensor(labels)
            labels = labels.to(device)
            outputs = model(**inputs)
            #ic(outputs.multimodal_output.pooler_output)
            #ic(outputs.multimodal_output.pooler_output.shape)
            
            classifier_inputs = outputs.multimodal_output.pooler_output.to(device)

            logits = classification_head(classifier_inputs)

            predicted = logits.argmax(dim=1).float()
            ic(predicted,labels)
            
            loss = criterion(predicted,labels)

            ## DDP code
            val_losses.append(accelerator.gather(loss))
            #ic(outputs.last_hidden_state)

            
            val_batch_corrects = len([i for i,
                                            j in zip(predicted, labels) if i == j])
            ic(val_batch_corrects)
            val_batch_corrects = torch.tensor(val_batch_corrects).to(device)

            val_corrects.append(accelerator.gather(val_batch_corrects))


            label_size = torch.tensor(len(labels)).to(device)
            gathered_sizes = accelerator.gather(label_size)
            val_size.append(gathered_sizes)


        total_val_size = torch.sum(torch.cat(val_size)).item()
        total_corrects = torch.sum(torch.cat(val_corrects)).item()

        ic(device,total_corrects,total_val_size)
        val_loss = torch.sum(torch.cat(val_losses)).item() / len(val_dataloader)

        val_accuracy =  total_corrects / total_val_size


        return val_loss,val_accuracy



def run_ddp_accelerate(args):
    EPOCHS = args.num_epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VAL_BATCH_SIZE = args.val_batch_size
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    LEARNING_RATE = args.learning_rate
    CHECKPOINT_DIR = args.checkpoint_dir
    EXPERIMENT_NAME = args.experiment_name
    WANDB_STATUS = args.wandb_status
    MACHINE_TYPE="ddp"
    LR_FLAG=args.lr
    ACCUMULATION_STEPS=args.accumulation_steps

    LR_FLAG=False
    if args.lr == 1: LR_FLAG=True 

    accelerator = Accelerator(gradient_accumulation_steps=ACCUMULATION_STEPS)

    if accelerator.is_main_process:
        wandb.init(project='CircuitBLIP',
                config = {
                'learning_rate':LEARNING_RATE,
                'epochs':EPOCHS,
                'train_batch_size': TRAIN_BATCH_SIZE,
                'val_batch_size':VAL_BATCH_SIZE
                },mode=WANDB_STATUS)
        wandb.run.name = EXPERIMENT_NAME

    
    # Record start time
    start_time = time.time()

    
    log_file = os.path.join(CHECKPOINT_DIR,'statistics.log')
    ic(log_file)
    if not os.path.exists(log_file):
        open(log_file, 'w+').close()
        print("log file created")
    else:
        print("log file exists")

    logging.basicConfig(filename=log_file,level=logging.INFO,filemode="a") # #,
    logger = logging.getLogger("mylogger")
    
   

    # tokenizer=BertTokenizer.from_pretrained("bert-base-uncased",padding="max_length",max_length=512)
    # model = flava_model_for_classification(num_classes=2)
    tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
    #model = FlavaMultimodalModel.from_pretrained("facebook/flava-full")
    model = FlavaModel.from_pretrained("facebook/flava-full")
    processor = AutoProcessor.from_pretrained("facebook/flava-full")

    train_dataset = FBHMDataset(root_dir=TRAIN_DIR,split='train')
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=TRAIN_BATCH_SIZE, collate_fn= collate_fn,pin_memory=False)

    val_dataset = FBHMDataset(root_dir=VAL_DIR, split='dev')
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=VAL_BATCH_SIZE, collate_fn= collate_fn,pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    #optimizer = bnb.optim.Adam8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
   
    if accelerator.is_main_process:
        wandb.watch(model)
        
    device = accelerator.device
    ic(device)
    criterion = torch.nn.CrossEntropyLoss()
   
    if LR_FLAG:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3, num_training_steps=10)
    else:
        scheduler = ""
        

    criterion = nn.BCELoss()

    train_dataloader, val_dataloader, model, optimizer,scheduler = accelerator.prepare(
        train_dataloader,val_dataloader, model, optimizer,scheduler
    )

    starting_name = 'checkpoint'
    stats_name = 'statistics'
    prev_accuracy = []
    prev_epoch = []
    best_accuracy = [0, 0.0]

    file_list = os.listdir(CHECKPOINT_DIR)
    ic(file_list)
    START = 0

    if file_list:
        checkpoint_files = [filename for filename in file_list if filename.startswith(starting_name)]
        #stat_files = [filename for filename in file_list if filename.startswith(stats_name)]
        ic(checkpoint_files)
        # Get latest checkpoint
        if checkpoint_files:
            checkpoint_files.sort()
            LATEST_CHECKPOINT=checkpoint_files[-1]
            START = int(checkpoint_files[-1].split(".pth")[0][-1:])+1
            ic("Previous checkpoint found",LATEST_CHECKPOINT,START)   
    
    stats = os.path.join(CHECKPOINT_DIR,'statistics.log')

    if os.path.exists(stats):
        with open(stats) as f:
            lines = f.readlines()
            #ic(lines)
            for line in lines:
                if 'Val accuracy' in line:
                    eid =line.index('Epoch') + 6
                    aid =line.index('Val accuracy') + 15

                    prev_epoch.append(int(line[eid]))
                    prev_accuracy.append(float(line[aid:aid+6]))
                    #ic(prev_accuracy,prev_epoch)
    ic(prev_accuracy,prev_epoch)

    if prev_accuracy:
        max_index, max_val = max(enumerate(prev_accuracy),key=lambda x:x[1])
        ic(max_val,max_index)
        best_accuracy = [prev_epoch[max_index],max_val]
        START = prev_epoch[-1] +1
        ic("Previous best accuracy found",best_accuracy[1],"best epoch",best_accuracy[0])
        chk_file = os.path.join(CHECKPOINT_DIR,'checkpoint_0' + str(max_index) + '.pth')
        ic(chk_file)
        model, optimizer,_= load_ckp(chk_file, model, optimizer)
    else:
        # Remove any checkpoints 
        for fname in os.listdir(CHECKPOINT_DIR):
            if fname.startswith("checkpoint"):
                os.remove(os.path.join(CHECKPOINT_DIR,fname))
                ic("removed checkpoint",fname)
            START=0

    ic("Running from epoch",START,"to epoch",EPOCHS)

    # START=0
    # EPOCHS=1
    # ic(START)
   
    for epoch in range(START,EPOCHS):
        start_time_epoch = time.time()

        train_loss,train_accuracy = train(epoch,model,train_dataloader,criterion,tokenizer,processor,optimizer,scheduler,device,accelerator,MACHINE_TYPE,ACCUMULATION_STEPS,LR_FLAG)
        val_loss, val_accuracy = evaluate(epoch,model,val_dataloader,criterion,tokenizer,processor,device,accelerator,MACHINE_TYPE,ACCUMULATION_STEPS,LR_FLAG)
             
        logger.info(f'Epoch {epoch} Train loss : {train_loss} Train accuracy : {train_accuracy}')
        logger.info(f'Epoch {epoch} Val loss : {val_loss} Val accuracy : {val_accuracy}')
        
        ic(epoch,train_loss,val_loss,train_accuracy,val_accuracy)
        
        if accelerator.is_main_process:
            ic("True")
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'epoch': epoch+1
            })      

            #Save model per epoch
            save_obj = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        #'config': model.config,
                        'epoch': epoch,
                    } 

            if epoch == 0:   
                ic("Saving 0 epoch checkpoint")
                torch.save(save_obj, os.path.join(
                    CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))
                best_accuracy[0] = epoch 
                best_accuracy[1] = float(val_accuracy)

            elif val_accuracy > best_accuracy[1] and epoch > 0:
                os.remove(os.path.join(CHECKPOINT_DIR,
                        'checkpoint_%02d.pth' % best_accuracy[0]))
                ic("old checkpoint removed")
                best_accuracy[0] = epoch
                best_accuracy[1] = float(val_accuracy)
                torch.save(save_obj, os.path.join(
                    CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))
            else:
                pass

            #Record end time
            end_time_epoch = time.time()
            run_time_epoch = end_time_epoch - start_time_epoch
            logger.info(f'Epoch {epoch} run time: {run_time_epoch:.2f} seconds')

    wandb.finish()
    ic("Completed training")


    
if __name__ == "__main__":

    #wandb.login(key='ce18e8ae96d72cd78a7a54de441e9657bc0a913d')
    parser = argparse.ArgumentParser()
    #set_seed(42)

    parser.add_argument('--num_epochs', default=1,
                        type=int, help='number of epochs')
    parser.add_argument('--learning_rate', default=2e-5,
                        type=int, help='Learning rate')
    parser.add_argument('--train_batch_size', default=2,
                        type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=2,
                        type=int, help='val batch size')
    parser.add_argument('--test_batch_size', default=2,
                        type=int, help='test batch size')
    parser.add_argument('--train_dir', help='train directory')
    parser.add_argument('--val_dir', help='Val directory')
    parser.add_argument('--checkpoint_dir', help='Val  directory')
    parser.add_argument('--experiment_name', help='exp name')
    parser.add_argument('--wandb_status',default='online', help='wandb set to online for online sync else disabled')
    parser.add_argument('--accumulation_steps',type=int,default=0,help="acc steps")
    parser.add_argument('--lr',type=int,default=0,help="lr")

    args = parser.parse_args()

    run_ddp_accelerate(args)


    
    