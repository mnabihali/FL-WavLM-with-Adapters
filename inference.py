import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import LibriSpeechDataset, DataCollatorCTCWithPadding
from modeling import AdaWavLMForCTC
from transformers import WavLMForCTC
from brain_update import inference
from distutils.util import strtobool
from datasets import load_metric
import sys, os
sys.path.append(os.pardir)
import argparse

##########################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--run_name', type=str, default='sample_run')
parser.add_argument('--use_skip', type=strtobool, default=False)
parser.add_argument('--use_adapter_fc', type=strtobool, default=True)
parser.add_argument('--use_adapter_norm', type=strtobool, default=True)
parser.add_argument('--eadapter_act', default='gelu')
parser.add_argument('--ladapter_act', default='gelu')
parser.add_argument('--lada_emb_size', type=int, default=512)
parser.add_argument('--eada_emb_size', type=int, default=256)
parser.add_argument('--train_encada', type=strtobool, default=False)
parser.add_argument('--train_eadapter', type=strtobool, default=False)
parser.add_argument('--use_adapter_ff', type=strtobool, default=True)
parser.add_argument('--use_adapter_attn', type=strtobool, default=True)
parser.add_argument('--adapter_init_std', type=float, default=1e-3)

parser.add_argument('--classifier_lr', type=float, default=1e-3)
parser.add_argument('--encoder_lr', type=float, default=1e-4)
parser.add_argument('--ladapter_lr', type=float, default=1e-3)
parser.add_argument('--eadapter_lr', type=float, default=1e-3)

parser.add_argument('--train_encoder', type=strtobool, default=False)
parser.add_argument('--weighted_sum', type=strtobool, default=False)
parser.add_argument('--train_lawithea', type=strtobool, default=False)

args = parser.parse_args()


##########################################################################

test_set = LibriSpeechDataset('librispeech_dataset_test.csv')
processor = test_set.processor
collator = DataCollatorCTCWithPadding(processor=processor, padding=True) 
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, collate_fn=collator, shuffle=False, num_workers=1, pin_memory=True)

##########################################################################

if args.train_encoder:
    model_config = {'ctc_loss_reduction':'mean',
                    'pad_token_id':processor.tokenizer.pad_token_id,
                    }
    learning_rate = {
            'classifier':args.classifier_lr,
            'encoder':args.encoder_lr,
        }
        
elif args.weighted_sum:
    model_config = {'ctc_loss_reduction':'mean',
                    'pad_token_id':processor.tokenizer.pad_token_id,
                    'use_adapter_to_output':True,
                    'adapter_to_output_layer_size': {str(i):args.lada_emb_size for i in range(0,12)},
                    'use_adapter_to_output_weighted_sum':True,
                    'use_adapter_fc':False,
                    'use_upsampling':False,
                    'use_residual':False,
                    'ladapter_act': None,
                    'use_adapter_norm':False,
                    }
    learning_rate = {
            'classifier':args.classifier_lr,
            'adapter_layer_weights':args.ladapter_lr, 
            'layer_norm':args.ladapter_lr,
            }
            
elif args.train_encada:
    model_config = {'ctc_loss_reduction':'mean',
                    'pad_token_id':processor.tokenizer.pad_token_id,
                    'adapter_embedding_size': {str(i):args.eada_emb_size for i in range(0,12)},
                    'eadapter_act': None if args.eadapter_act=='None' else args.eadapter_act,
                    'use_adapter_ff': args.use_adapter_ff,
                    'use_adapter_attn': args.use_adapter_attn,
                    'adapter_init_std': args.adapter_init_std
                    }
    learning_rate = {
            'classifier':args.classifier_lr,
            'adapter_ff':args.eadapter_lr,
            'adapter_attn':args.eadapter_lr,          
            'layer_norm':args.eadapter_lr,
            }

elif args.train_lawithea:
    model_config = {'ctc_loss_reduction': 'mean',
                    'pad_token_id': processor.tokenizer.pad_token_id,
                    'use_adapter_to_output': True,
                    'use_adapter_to_output_weighted_sum': True,
                    'adapter_to_output_layer_size': {str(i): args.lada_emb_size for i in range(0, 12)},
                    'use_adapter_fc': args.use_adapter_fc,
                    'use_upsampling': args.use_skip,
                    'use_residual': args.use_skip,
                    'use_adapter_norm': args.use_adapter_norm,
                    'adapter_embedding_size': {str(i): args.eada_emb_size for i in range(0, 11)},
                    'ladapter_act': None if args.ladapter_act == 'None' else args.ladapter_act,
                    'eadapter_act': None if args.eadapter_act == 'None' else args.eadapter_act,
                    'use_adapter_ff': True,
                    'use_adapter_attn': False,
                    'adapter_init_std': args.adapter_init_std
                    }
    learning_rate = {
        'classifier': args.classifier_lr,
        'adapter_to_output': args.ladapter_lr,
        'adapter_layer_weights': args.ladapter_lr,
        'adapter_ff': args.eadapter_lr,
        'layer_norm': args.eadapter_lr,
    }
else:
    model_config = {'ctc_loss_reduction': 'mean',
                    'pad_token_id': processor.tokenizer.pad_token_id,
                    'use_adapter_to_output': True,
                    'adapter_to_output_layer_size': {str(i): args.lada_emb_size for i in range(0, 12)},
                    'use_adapter_to_output_weighted_sum': True,
                    'use_adapter_fc': args.use_adapter_fc,
                    'use_upsampling': args.use_skip,
                    'use_residual': args.use_skip,
                    'ladapter_act': None if args.ladapter_act == 'None' else args.ladapter_act,
                    'use_adapter_norm': args.use_adapter_norm,
                    }
    learning_rate = {
        'classifier': args.classifier_lr,
        'adapter_to_output': args.ladapter_lr,
        'adapter_layer_weights': args.ladapter_lr,
        'layer_norm': args.ladapter_lr,
    }
    
config = {
    "pretrained_model": 'microsoft/wavlm-base-plus',
    "dataset": 'Librispeech',
    "epochs": 5,
    "model_config": model_config,
    "learning_rate": learning_rate,
    'optimizer': 'Adam',
    "scheduler": {'type': 'StepLR', 'step': 25, 'gamma': 0.3} if args.train_encada and args.use_steplr else {
        'type': 'LambdaLR', 'param': {'alpha': 0.20, 'beta': 0.03, 'start': 10, 'end': 1.0, 'scale': 10}},
}

num_epochs = config['epochs']
learning_rate = config['learning_rate']
sc_setting = config['scheduler']
pretrained_model = config['pretrained_model']




if args.train_encoder:
    model = WavLMForCTC.from_pretrained(pretrained_model, **model_config)
else:
    model = AdaWavLMForCTC.from_pretrained(pretrained_model, **model_config)

##########################################################################


model.load_state_dict(torch.load("/stek/mohamed/Wavlm_adapt_FL/FL_dp_107/wavlm-dp-adapt-round-10.pth"))   # Don't forget to add the path of you model here

metric = load_metric('wer')



test_loss, test_wer = inference(model, processor, test_loader, metric)
