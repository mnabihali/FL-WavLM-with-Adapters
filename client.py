from collections import OrderedDict
import psutil
import numpy as np
import argparse
import torch
import flwr as fl
import random
import gc
import glob
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import time
from distutils.util import strtobool
from dataset import load_dataloaders
from transformers import WavLMForCTC
from modeling import AdaWavLMForCTC
from brain_update import train_asr, fix_seed
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (Code, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns, GetParametersRes, Status,
                         Scalar, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays, )
from typing import Dict, List, Optional, Tuple, Callable, Union

#################################################################################

# Argument Parser
sys.path.append(os.pardir)
fix_seed(42)
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

#################################################################################

# Loading Librispeech Dataset by speaker
train_csv_files_directory = "train_files"  # Directory containing the CSV files
train_loaders, processor = load_dataloaders(train_csv_files_directory)
print(len(train_loaders))

dev_csv_files_directory = "dev_files"  # Directory containing the CSV files
dev_loaders, processor = load_dataloaders(dev_csv_files_directory)
print(len(dev_loaders))

#################################################################################

# Load Model & Model Config & Paramters Count

if args.train_lawithea:
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


net = AdaWavLMForCTC.from_pretrained(pretrained_model, **model_config)
    
###############################################################################

# Counting # of Parameters
down_param = []
layernorm_param = []
encoder_param = []
adapter_ff_param = []
adapter_attn_param = []
adapter_to_output_param = []
adapter_to_output_layer_weights_param = []
pcount = 0
adapcount = 0
flag = True

if args.train_encoder:
    layer_names = [str(i) for i in range(0, 12)]
elif args.weighted_sum:
    layer_names = [str(i) for i in range(12)]
elif args.train_encada:
    layer_names = ['layers.' + k for k in model_config["adapter_embedding_size"].keys()]
else:
    layer_names = ['layers.' + k for k in model_config["adapter_to_output_layer_size"].keys()]

for name, param in net.named_parameters():
    for layer in layer_names:
        if layer in name:
            flag = True
            break
        else:
            flag = False

    if 'lm_head' in name:
        print('down_param: ', name)
        pcount += param.numel()
        down_param.append(param)

    elif 'adapter_to_output_layer_weights' in name:
        adapter_to_output_layer_weights_param.append(param)
        print('adapter_to_output_layer_weights: ', name)
        pcount += param.numel()
        adapcount += param.numel()

    elif 'encoder.layers' in name and 'layer_norm' in name and flag and not args.train_encoder:
        layernorm_param.append(param)
        print('layer_norm: ', name)
        pcount += param.numel()

    elif 'adapter_layer_ff' in name:
        adapter_ff_param.append(param)
        print('enc_adapter_ff: ', name)
        pcount += param.numel()
        adapcount += param.numel()

    elif 'adapter_layer_attn' in name:
        adapter_attn_param.append(param)
        print('enc_adapter_attn: ', name)
        pcount += param.numel()
        adapcount += param.numel()

    elif 'adapter_to_output' in name:
        adapter_to_output_param.append(param)
        print('adapter_output: ', name)
        pcount += param.numel()
        adapcount += param.numel()

    elif 'encoder.layers' in name and flag and args.train_encoder:
        encoder_param.append(param)
        pcount += param.numel()
        print('encoder: ', name)

    else:
        print('frozen: ', name)
        param.requires_grad = False

print('\ncount of parameters: ', pcount, '\n')
print('\ncount of adapter_parameters: ', adapcount, '\n')

config.update({'num_params (1e7)': pcount / 1e7})
config.update({'num_adapter_params (M)': adapcount / 1e6})



if args.train_lawithea:
    optimizer = torch.optim.Adam([
        {'params': down_param, 'lr': learning_rate['classifier']},
        {'params': adapter_ff_param, 'lr': learning_rate['adapter_ff']},
        {'params': adapter_to_output_layer_weights_param, 'lr': learning_rate['adapter_layer_weights']},
        {'params': adapter_to_output_param, 'lr': learning_rate['adapter_to_output']},
        {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
    ])

else:
    optimizer = torch.optim.Adam([
        {'params': down_param, 'lr': learning_rate['classifier']},
        {'params': adapter_to_output_layer_weights_param, 'lr': learning_rate['adapter_layer_weights']},
        {'params': adapter_to_output_param, 'lr': learning_rate['adapter_to_output']},
        {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
    ])

if args.train_encada and args.use_steplr:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sc_setting['step'], gamma=sc_setting['gamma'])
else:
    hyparam = sc_setting['param']

    def func(epoch):
        alpha = hyparam['alpha']
        beta = hyparam['beta']
        start = hyparam['start']
        end = hyparam['end']
        scale = hyparam['scale']
        warmup = np.linspace(start, num_epochs, int(num_epochs * alpha)) / num_epochs
        stag = np.ones(int(num_epochs * beta))
        decay = np.linspace(num_epochs, end, int(num_epochs * (1 - alpha - beta) + 1)) / np.linspace(num_epochs,
                                                                                                     num_epochs * scale,
                                                                                                     int(num_epochs *
                                                                                                         (1 - alpha -
                                                                                                          beta) + 1))
        steps = np.concatenate([warmup, stag, decay], axis=-1)
        return steps[epoch - 1]


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
    
###############################################################################

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )
    net.load_state_dict(state_dict, strict=True)

###############################################################################

class CustomStrategyAdam(fl.server.strategy.FedOpt):
    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Parameters,
            eta: float = 1e-1,
            eta_l: float = 1e-1,
            beta_1: float = 0.9,
            beta_2: float = 0.99,
            tau: float = 0.1,
    ) -> None:

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAdam(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        sample_size = random.randint(min_num_clients, sample_size)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Create custom configs
        n_clients = len(clients)

        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        # print("CONFIG.  sample:", sample_size,"min_num_clients:", min_num_clients, "n_clients:",n_clients)
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

        # Adam
        delta_t: NDArrays = [
            x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y)
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        self.current_weights = new_weights

        params_dict = zip(net.state_dict().keys(), self.current_weights)
        state_dict = OrderedDict({k: torch.Tensor(np.array(v)) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        mu = server_round + 36
        torch.save(net.state_dict(), f"FL_adam_again/wavlm-adapt-adam-round-{mu}.pth")

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated

'''
class CustomStrategy(fl.server.strategy.FedAvg):

    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self) -> str:
        return "FedCustom"

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        sample_size = random.randint(min_num_clients, sample_size)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)

        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                      failures: List[BaseException],
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:  # Optional [fl.common.Parameters] :
        # #Tuple[Optional[Parameters], Dict[str, Scalar]]

        if not results:
            return None, {}

        if self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        weights = aggregate(weights_results)

        if weights is not None:

            params_dict = zip(net.state_dict().keys(), weights)
            state_dict = OrderedDict({k: torch.Tensor(np.array(v)) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            # nu = server_round + 200  # modification
            torch.save(net.state_dict(), f"FL/wavlm-adapt-round-{server_round}.pth")  # trained_models  #server_round
            new_parameters = get_parameters(net)

            del results, weights_results
            gc.collect()
            return ndarrays_to_parameters(new_parameters), {}
        else:
            print(f"returning None weights, something went wrongh during aggregation..... !!!!!!!!!!!!!!!")

            del results, weights_results
            gc.collect()
            return ndarrays_to_parameters(weights), {}
'''

###############################################################################

class Clientasr(fl.client.Client):
    def __init__(self, cid, net, dataloader_dict):
        self.cid = cid
        self.net = net
        self.dataloader_dict = dataloader_dict

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.cid} get parameters]")
        ndarrays: List[np.ndarray] = get_parameters(self.net)
        parameters = ndarrays_to_parameters(ndarrays)
        status = Status(code=Code.OK, message='Success')
        torch.cuda.empty_cache()
        return GetParametersRes(status=status, parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        print('Starting Training')
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_parameters(self.net, ndarrays_original)

        epoch_loss, num_samples = train_asr(self.net, self.dataloader_dict, processor, optimizer, scheduler, num_epochs)
        
        ndarrays_updated = get_parameters(self.net)
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)
        status = Status(code=Code.OK, message='Success')
        metrics = {'Loss': epoch_loss}
        print('Traing Success and send parameters')
        print('Hello Mohamed Hello')
        torch.cuda.empty_cache()
        return FitRes(status=status, parameters=parameters_updated, num_examples=num_samples, metrics=metrics)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print('Starting Training')
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_parameters(self.net, ndarrays_original)

        epoch_loss, num_samples, epoch_wer = train_asr(self.net, processor, self.dataloader_dict, optimizer, scheduler, num_epochs, val_interval=5)
        
        ndarrays_updated = get_parameters(self.net)
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)
        status = Status(code=Code.OK, message='Success')
        metrics = {'Loss': epoch_loss}
        torch.cuda.empty_cache()
        return  EvaluateRes(status=status, parameters=parameters_updated, num_examples=num_samples, metrics=metrics)


def client_fn(cid) -> Clientasr:
    trainloader = train_loaders[int(cid)]
    #devloader = dev_loaders[int(cid)]
    dataloaders_dict = {'train': trainloader, 'val': trainloader}
    return Clientasr(cid, net, dataloaders_dict)


ram_memory = 16_000 * 1024 * 1024

client_resources = {"num_cpus":1, "num_gpus":1}


state_dict = torch.load("/stek/mohamed/Wavlm_adapt_FL/FL_adam_again/wavlm-adapt-adam-round-36.pth")
net.load_state_dict(state_dict, strict=True)
ndarrays = get_parameters(net)

my_strategy = CustomStrategyAdam(fraction_fit=0.3,
                             fraction_evaluate=0,
                             min_fit_clients=2,
                             min_evaluate_clients=2,
                             min_available_clients=2,
                             initial_parameters = fl.common.ndarrays_to_parameters(ndarrays),
                             )

fl.simulation.start_simulation(client_fn=client_fn,
                               num_clients=251,
                               config=fl.server.ServerConfig(num_rounds=200),
                               strategy=my_strategy,
                               ray_init_args={
                                   "include_dashboard": False,  # we need this one for tracking
                                   "num_cpus": 2,
                                   "num_gpus": 1,
                                   "_memory": ram_memory,
                                   "object_store_memory": 10 ** 9,
                               },
                               client_resources=client_resources)
