import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import os
import time
import random
import tqdm
from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from jiwer import wer


def train_asr(model, dataloaders_dict, processor, optimizer, scheduler, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    torch.backends.cudnn.benchmark = True

    
    opt_flag = (type(optimizer) == list)
    sc_flag = (type(scheduler) == list)

    for epoch in range(num_epochs):
        print('num_epochs', num_epochs)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            epoch_loss = 0.0
            #epoch_wer = 0.
            #epoch_preds_str = []
            #epoch_labels_str = []

            for step, inputs in enumerate(dataloaders_dict[phase]):
            
                num_samples = len(dataloaders_dict[phase].dataset)
                minibatch_size = inputs['input_values'].size(0)
                labels_ids = inputs['labels']
                inputs = inputs.to(device)

                if opt_flag:
                    for opt in optimizer:
                        opt.zero_grad()
                else:
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(**inputs)
                    del inputs
                    loss = outputs.loss.mean(dim=-1)
                    #print('loss is', loss)
                    preds_ids = torch.argmax(outputs.logits, dim=-1)
                    preds_str = processor.batch_decode(preds_ids)
                    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
                    labels_str = processor.batch_decode(labels_ids, group_tokens=False)
                    #print('labels_str',labels_str)
                    #WER = wer(preds_str, labels_str)
                    #epoch_preds_str += preds_str
                    #epoch_labels_str += labels_str

                if phase == 'train':
                    loss.backward()
                    if opt_flag:
                        for opt in optimizer:
                            opt.step()
                    else:
                        optimizer.step()
                    loss_log = loss.item()
                    del loss

                epoch_loss += loss_log * minibatch_size


            #epoch_wer = wer(epoch_preds_str, epoch_labels_str)
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

            if phase == 'train':
                if scheduler:
                    if sc_flag:
                        for sc in scheduler:
                            sc.step()
                    else:
                        scheduler.step()
                print('Epoch {}/{} | {:^5} |  Loss: {:.4f}'.format(epoch + 1, num_epochs, phase, epoch_loss))

            else:
                print('Epoch {}/{} | {:^5} |  Loss: {:.4f}'.format(epoch + 1, num_epochs, phase, epoch_loss))

    return epoch_loss, num_samples
    
    
def inference(model, processor, dataloader, metric):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    epoch_loss = 0.0
    epoch_wer = 0.0
    epoch_preds_str = []
    epoch_labels_str = []

    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            print('Step', step)
            minibatch_size = inputs['input_values'].size(0)
            labels_ids = inputs['labels']
            inputs = inputs.to(device)

            outputs = model(**inputs)
            loss = outputs.loss.mean(dim=-1)
            preds_ids = torch.argmax(outputs.logits, dim=-1)
            preds_str = processor.batch_decode(preds_ids)
            labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
            labels_str = processor.batch_decode(labels_ids, group_tokens=False)
            print('preds_str', preds_str)
            print('labels_str', labels_str)
            wer = metric.compute(predictions=preds_str, references=labels_str)
            epoch_preds_str += preds_str
            epoch_labels_str += labels_str

            loss_log = loss.item()
            epoch_loss += loss_log * minibatch_size

    epoch_loss = epoch_loss / len(dataloader.dataset)
    epoch_wer = metric.compute(predictions=epoch_preds_str, references=epoch_labels_str)

    print('Validation |  Loss: {:.4f} WER: {:.4f}'.format(epoch_loss, epoch_wer))

    return epoch_loss, epoch_wer

    
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
