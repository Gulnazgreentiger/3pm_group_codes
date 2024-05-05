# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:57:50 2024

@author: Prakash
"""
# Install the torchmetrics package
#!pip install torchmetrics

# Import the StepLR class
from torch.optim.lr_scheduler import StepLR

# Restart the notebook
# ... Your other code here ...
CONSTANTS = ["orography", "land_sea_mask", "slt", "lattitude", "longitude"]
# Standard library
from typing import Callable, List, Optional, Tuple, Union

# Third party
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

#%%#%% Testing climatology

TRANSFORMS_REGISTRY = {}

def register(name):
    def decorator(transform_cls):
        TRANSFORMS_REGISTRY[name] = transform_cls
        return transform_cls

    return decorator

# Define a mean and standard deviation for normalization
#mean = [1,2,3]
mean = 3
std = 1
#std = [1,2,1]

@register("climatology")
class Climatology(nn.Module):
    def __init__(self, clim, mean, std):
        super().__init__()
        self.norm = transforms.Normalize(mean, std)
        self.clim = clim  # clim.shape = [C,H,W]

    def forward(self, x):
        # x.shape = [B,T,C,H,W]
        yhat = self.norm(self.clim).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        # yhat.shape = [B,C,H,W]
        return yhat

# Example usage
# Create an instance of the Climatology model with some sample climatology data
#climatology_data = torch.randn(3, 3, 32, 32)  # Example climatology data with shape [C,H,W]
climatology_data = torch.ones(3, 4, 4) 
clim_model = Climatology(clim=climatology_data, mean=mean, std=std)

# Assuming you have some input data x
input_data = torch.full((2, 5, 3, 4, 4),2)  # Example input data with shape [B,T,C,H,W]

# Forward pass through the model
output = clim_model(input_data)
## clim_model averages the time dim
print(output.shape)  # Output shape: [2, 3, 32, 32]




#%%
# Standard library
from typing import Iterable, Optional
@register("persistence")
class Persistence(nn.Module):
    def __init__(self, channels: Optional[Iterable[int]] = None):
        """
        :param channels: The indices of the channels to forward from the input
            to the output.
        """
        super().__init__()
        self.channels = channels

    def forward(self, x):
        # x.shape = [B,T,in_channels,H,W]
        if self.channels:
            yhat = x[:, -1, self.channels]
        else:
            yhat = x[:, -1]
        # yhat.shape = [B,out_channels,H,W]
        return yhat

## Persistence function just uses the last timestep. 
## There are 2 ways: whether to specify channels or not.
## usage   
B = 2  # Batch size
T = 5  # Number of time steps
in_channels = 3  # Number of input channels
H = 32  # Height
W = 32  # Width

# Create some sample input data
x = torch.randn(B, T, in_channels, H, W)

# Create an instance of the Persistence model without specifying channels
persistence_model_no_channels_specified = Persistence()

# Forward pass through the model
output_no_channels_specified = persistence_model_no_channels_specified(x)
print("Output shape without specifying channels:", output_no_channels_specified.shape)
## In the output, no time dim coz we used just last timestep. 
## All channels are present coz we did not specify channels.


# Create an instance of the Persistence model by specifying channels
channels = [0, 2]  # let's specify 1st and third channels
persistence_model_with_channels = Persistence(channels)

# Forward pass through the model
output_with_channels = persistence_model_with_channels(x)
print("Output shape with specified channels:", output_with_channels.shape)
## In this output, again no time dim coz we used just last timestep
## Only 2 channels out of 3 coz we specified only 2. 
#%% LitModule

class LitModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        train_loss: Callable,
        val_loss: List[Callable],
        test_loss: List[Callable],
        train_target_transform: Optional[Callable] = None,
        val_target_transforms: Optional[List[Union[Callable, None]]] = None,
        test_target_transforms: Optional[List[Union[Callable, None]]] = None,
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss
        self.train_target_transform = train_target_transform
        if val_target_transforms is not None:
            if len(val_loss) != len(val_target_transforms):
                raise RuntimeError(
                    "If 'val_target_transforms' is not None, its length must"
                    " match that of 'val_loss'. 'None' can be passed for"
                    " losses which do not require transformation."
                )
        self.val_target_transforms = val_target_transforms
        if test_target_transforms is not None:
            if len(test_loss) != len(test_target_transforms):
                raise RuntimeError(
                    "If 'test_target_transforms' is not None, its length must"
                    " match that of 'test_loss'. 'None' can be passed for "
                    " losses which do not rqeuire transformation."
                )
        self.test_target_transforms = test_target_transforms
        self.mode = "direct"

    def set_mode(self, mode):
        self.mode = mode

    def set_n_iters(self, iters):
        self.n_iters = iters

    def replace_constant(self, y, yhat, out_variables):
        for i in range(yhat.shape[1]):
            # if constant replace with ground-truth value
            if out_variables[i] in CONSTANTS:
                yhat[:, i] = y[:, i]
        return yhat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        #yhat = self.replace_constant(y, yhat, out_variables)
        if self.train_target_transform:
            yhat = self.train_target_transform(yhat)
            y = self.train_target_transform(y)
        losses = self.train_loss(yhat, y)
        loss_name = getattr(self.train_loss, "name", "loss")
        loss_dict = {}
        if losses.dim() == 0:  # aggregate loss only
            loss = losses
            loss_dict[f"train/{loss_name}:aggregate"] = loss
        else:  # per channel + aggregate
            for var_name, loss in zip(out_variables, losses):
                loss_dict[f"train/{loss_name}:{var_name}"] = loss
            loss = losses[-1]
            loss_dict[f"train/{loss_name}:aggregate"] = loss
        self.log_dict(
            loss_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=x.shape[0],
        )
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        self.evaluate(batch, "val")

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        if self.mode == "direct":
            self.evaluate(batch, "test")
        if self.mode == "iter":
            self.evaluate_iter(batch, self.n_iters, "test")

    def evaluate(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]], stage: str
    ):
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        #yhat = self.replace_constant(y, yhat, out_variables)
        if stage == "val":
            loss_fns = self.val_loss
            transforms = self.val_target_transforms
        elif stage == "test":
            loss_fns = self.test_loss
            transforms = self.test_target_transforms
        else:
            raise RuntimeError("Invalid evaluation stage")
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            if transforms is not None and transforms[i] is not None:
                yhat_ = transforms[i](yhat)
                y_ = transforms[i](y)
            losses = lf(yhat_, y_)
            loss_name = getattr(lf, "name", f"loss_{i}")
            if losses.dim() == 0:  # aggregate loss
                loss_dict[f"{stage}/{loss_name}:agggregate"] = losses
            else:  # per channel + aggregate
                for var_name, loss in zip(out_variables, losses):
                    name = f"{stage}/{loss_name}:{var_name}"
                    loss_dict[name] = loss
                loss_dict[f"{stage}/{loss_name}:aggregate"] = losses[-1]
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch[0]),
        )
        return loss_dict
#%%
from functools import wraps
Pred = Union[torch.FloatTensor, torch.DoubleTensor, torch.distributions.Normal]
def handles_probabilistic(metric):
    @wraps(metric)
    def wrapper(pred: Pred, *args, **kwargs):
        if isinstance(pred, torch.distributions.Normal):
            pred = pred.loc
        return metric(pred, *args, **kwargs)

    return wrapper

@handles_probabilistic
def mse(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    #error = (pred - target).square()
    error = (pred - target.unsqueeze(-1)) ** 2
    if lat_weights is not None:
        error = error * lat_weights
    per_channel_losses = error.mean([0, 2, 3])
    loss = error.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


# Assuming you have defined your neural network architecture
# class MyNet(nn.Module):
#     def __init__(self):
#         super(MyNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc = nn.Linear(in_features=64 * 16 * 16, out_features=10)
# Instantiate your neural network
# net = MyNet()
# net.train() 

clim_model = Climatology(clim=climatology_data, mean=mean, std=std)
# Define optimizer and learning rate scheduler
#optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer = None
lr_scheduler = None
#lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
# Define loss functions for training, validation, and test
train_loss_fn = mse
val_loss_fn = mse
test_loss_fn = mse

#device = torch.device("cuda" if torch.cuda.is_available() else "auto")
device = torch.device('cpu')
net = clim_model.to(device)

# Instantiate LitModule with the defined components
lit_module = LitModule(
    net=clim_model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    train_loss=train_loss_fn,
    val_loss=val_loss_fn,
    test_loss=test_loss_fn,
)

from torch.utils.data import DataLoader, IterableDataset

tmp = DataLoader(individual_data_iter, batch_size=3, collate_fn=collate_fn)
for inp,out,inV,OutV in tmp:
    print(inp.shape,out.shape,inV,OutV)

for batch in tmp:
    lit_module.training_step(batch, 0)
# for sample in individual_data_iter:
#         print(sample)

#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:51:33 2024

@author: Prakash
"""

import glob
import os

## Third party
import numpy as np
import xarray as xr
import netCDF4 as nc
from tqdm import tqdm
import random

#%%
NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
    "vorticity": "vo",
    "potential_vorticity": "pv",
    "total_cloud_cover": "tcc",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

DEFAULT_PRESSURE_LEVELS = [50, 250, 500, 600, 700, 850, 925]

CONSTANTS = ["orography", "land_sea_mask", "slt", "lattitude", "longitude"]
#%% 
HOURS_PER_YEAR = 24 ## LOOK AT THE SHARD WITH SMALL DATA
def nc2np(path, variables, years, save_dir, partition, num_shards_per_year):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
    climatology = {}
   
    for year in tqdm(years):
        np_vars = {}

        # non-constant fields
        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            ds = xr.open_mfdataset(
                ps, combine="by_coords", parallel=True
            )  # dataset for a single variable
            code = NAME_TO_VAR[var]
            print(ds[code].shape)

            if len(ds[code].shape) == 3:  # surface level variables
                ds[code] = ds[code].expand_dims("val", axis=1)
                # remove the last 24 hours if this year has 366 days
                if code == "tp":  # accumulate 6 hours and log transform
                    tp = ds[code].to_numpy()
                    tp_cum_6hrs = np.cumsum(tp, axis=0)
                    tp_cum_6hrs[6:] = tp_cum_6hrs[6:] - tp_cum_6hrs[:-6]
                    eps = 0.001
                    tp_cum_6hrs = np.log(eps + tp_cum_6hrs) - np.log(eps)
                    np_vars[var] = tp_cum_6hrs[-HOURS_PER_YEAR:]
                else:
                    np_vars[var] = ds[code].to_numpy()[-HOURS_PER_YEAR:]

                if partition == "train":
                    # compute mean and std of each var in each year
                    var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                    var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
                    if var not in normalize_mean:
                        normalize_mean[var] = [var_mean_yearly]
                        normalize_std[var] = [var_std_yearly]
                    else:
                        normalize_mean[var].append(var_mean_yearly)
                        normalize_std[var].append(var_std_yearly)

                clim_yearly = np_vars[var].mean(axis=0)
                if var not in climatology:
                    climatology[var] = [clim_yearly]
                else:
                    climatology[var].append(clim_yearly)

        assert HOURS_PER_YEAR % num_shards_per_year == 0
        num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            print("Shard:", shard_id)
            for k, v in sharded_data.items():
                print(f"Variable: {k}, Shape: {v.shape}")
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                **sharded_data,
           )

    if partition == "train":
        for var in normalize_mean.keys():
            #if not constants_are_downloaded or var not in constant_fields:
                normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
                normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            #if not constants_are_downloaded or var not in constant_fields:
                mean, std = normalize_mean[var], normalize_std[var]
                # var(X) = E[var(X|Y)] + var(E[X|Y])
                variance = (
                    (std**2).mean(axis=0)
                    + (mean**2).mean(axis=0)
                    - mean.mean(axis=0) ** 2
                )
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean.mean(axis=0)
                normalize_mean[var] = mean
                if var == "total_precipitation":
                    normalize_mean[var] = np.zeros_like(normalize_mean[var])
                normalize_std[var] = std

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    np.savez(
        os.path.join(save_dir, partition, "climatology.npz"),
        **climatology,
    )
#%%
variables = ['2m_temperature','total_precipitation']
#path = 'C:/Users/Prakash/OneDrive - University of California, Davis/califo_work/seaonal_prediction/clim_learn_test/data/'
path = 'C:/Users/Prakash/OneDrive - University of California, Davis/califo_work/seaonal_prediction/clim_learn_test/data/'
save_dir = path + 'processed'
#partition = path + 'test'
nc2np(path, variables, range(1987, 2001), save_dir, 'train', 4)
nc2np(path,variables, [2017], save_dir, 'test',4)
nc2np(path,variables, [2018], save_dir, 'val',4)
#%% Read the output file to see the data
# Load the .npz file
data = np.load(save_dir + '/test' + '/2017_0.npz')

# Iterate over the keys of the data dictionary
for key in data.keys():
    print(f"The name of the array is '{key}'")

tmp = data['2m_temperature']
pr = data['total_precipitation']

## check shape
tmp.shape
pr.shape
#%% Now test NpyReader
import torch
from torch.utils.data import IterableDataset

class NpyReader(IterableDataset):
    def __init__(
        self,
        inp_file_list,
        out_file_list,
        variables,
        out_variables,
        shuffle=False,
    ):
        super().__init__()
        print("Length of inp_file_list:", len(inp_file_list))
        print("Length of out_file_list:", len(out_file_list))

        assert len(inp_file_list) == len(out_file_list)
        self.inp_file_list = [f for f in inp_file_list if "climatology" not in f]
        self.out_file_list = [f for f in out_file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle

    def __iter__(self):
        # if self.shuffle:
        #     self.inp_file_list, self.out_file_list = shuffle_two_list(
        #         self.inp_file_list, self.out_file_list
        #     )

        n_files = len(self.inp_file_list)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = n_files
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            num_shards = num_workers_per_ddp * world_size
            per_worker = n_files // num_shards
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for idx in range(iter_start, iter_end):
            path_inp = self.inp_file_list[idx]
            path_out = self.out_file_list[idx]
            inp = np.load(path_inp)
            if path_out == path_inp:
                out = inp
            else:
                out = np.load(path_out)
            yield {k: np.squeeze(inp[k], axis=1) for k in self.variables}, {
                k: np.squeeze(out[k], axis=1) for k in self.out_variables
            }, self.variables, self.out_variables
#%% Initialize the class
inp_lister_test = sorted(
    glob.glob(os.path.join(path, 'processed','test','*.npz')))

out_lister_test = sorted(
    glob.glob(os.path.join(path, 'processed','test','*.npz')))

    
tmp = NpyReader(
    inp_file_list=inp_lister_test,
    out_file_list=out_lister_test,
    variables = variables,
    out_variables=variables,
    shuffle=False,
    )
# print the outputs
for xIn, xOut, varIn, varOut in tmp:
    # for k, v in xIn.items():
    #     print(k,v.shape)
    print(varIn)
#%% Direct Forecasts
class DirectForecast(IterableDataset):
    def __init__(self, dataset, src, pred_range=6, history=3, window=6):
        super().__init__()
        self.dataset = dataset
        self.history = history
        if src == "era5":
            self.pred_range = pred_range
            self.window = window
        elif src == "mpi-esm1-2-hr":
            assert pred_range % 6 == 0
            assert window % 6 == 0
            self.pred_range = pred_range // 6
            self.window = window // 6

    def __iter__(self):
        for inp_data, out_data, variables, out_variables in self.dataset:
            inp_data = {
                k: torch.from_numpy(inp_data[k].astype(np.float32))
                .unsqueeze(0)
                .repeat_interleave(self.history, dim=0)
                for k in inp_data.keys()
            }
            out_data = {
                k: torch.from_numpy(out_data[k].astype(np.float32))
                for k in out_data.keys()
            }
            for key in inp_data.keys():
                for t in range(self.history):
                    inp_data[key][t] = inp_data[key][t].roll(-t * self.window, dims=0)

            last_idx = -((self.history - 1) * self.window + self.pred_range)

            inp_data = {
                k: inp_data[k][:, :last_idx].transpose(0, 1)
                for k in inp_data.keys()  # N, T, H, W
            }

            inp_data_len = inp_data[variables[0]].size(0)

            predict_ranges = torch.ones(inp_data_len).to(torch.long) * self.pred_range
            output_ids = (
                torch.arange(inp_data_len)
                + (self.history - 1) * self.window
                + predict_ranges
            )
            out_data = {k: out_data[k][output_ids] for k in out_data.keys()}
            yield inp_data, out_data, variables, out_variables    
    
#%% usage DirectForecast
tmp1 = DirectForecast(tmp,'era5', pred_range=1, history=2, window=1)    
 
# print the variables in input tmp
for xIn, xOut, varIn, varOut in tmp:
    print("In:")
    for k,v in xIn.items():  
       print(f"{k.ljust(20)}: {v[:,0,0]}")
    print("Out:")
    for k,v in xOut.items():  
       print(f"{k.ljust(20)}: {v[:,0,0]}")
    break

## same thing but for output tmp1
for xIn, xOut, varIn, varOut in tmp1:
    print("In:")
    for k,v in xIn.items():  
       print(f"{k.ljust(20)}: {v[:,:,0,0]}")
    print("Out:")
    for k,v in xOut.items():  
       print(f"{k.ljust(20)}: {v[:,0,0]}")
    break
#%%
class ContinuousForecast(IterableDataset):
    def __init__(
        self,
        dataset,
        random_lead_time=True,
        min_pred_range=6,
        max_pred_range=120,
        hrs_each_step=1,
        history=3,
        window=6,
    ):
        super().__init__()
        if not random_lead_time:
            assert min_pred_range == max_pred_range
        self.dataset = dataset
        self.random_lead_time = random_lead_time
        self.min_pred_range = min_pred_range
        self.max_pred_range = max_pred_range
        self.hrs_each_step = hrs_each_step
        self.history = history
        self.window = window

    def __iter__(self):
        for inp_data, out_data, variables, out_variables in self.dataset:
            inp_data = {
                k: torch.from_numpy(inp_data[k].astype(np.float32))
                .unsqueeze(0)
                .repeat_interleave(self.history, dim=0)
                for k in inp_data.keys()
            }
            out_data = {
                k: torch.from_numpy(out_data[k].astype(np.float32))
                for k in out_data.keys()
            }
            for key in inp_data.keys():
                for t in range(self.history):
                    inp_data[key][t] = inp_data[key][t].roll(-t * self.window, dims=0)

            last_idx = -((self.history - 1) * self.window + self.max_pred_range)

            inp_data = {
                k: inp_data[k][:, :last_idx].transpose(0, 1)
                for k in inp_data.keys()  # N, T, H, W
            }

            inp_data_len = inp_data[variables[0]].size(0)
            dtype = inp_data[variables[0]].dtype

            if self.random_lead_time:
                predict_ranges = torch.randint(
                    low=self.min_pred_range,
                    high=self.max_pred_range + 1,
                    size=(inp_data_len,),
                )
            else:
                predict_ranges = (
                    torch.ones(inp_data_len).to(torch.long) * self.max_pred_range
                )
            lead_times = self.hrs_each_step * predict_ranges / 100
            lead_times = lead_times.to(dtype)
            output_ids = (
                torch.arange(inp_data_len)
                + (self.history - 1) * self.window
                + predict_ranges
            )

            out_data = {k: out_data[k][output_ids] for k in out_data.keys()}
            yield inp_data, out_data, lead_times, variables, out_variables
#%% Usage Continuous Forecast
tmp2 = ContinuousForecast(tmp, random_lead_time=True, 
                          hrs_each_step=1, min_pred_range=1, max_pred_range=2, history=2,window=2)

## same thing but for output tmp2
for xIn, xOut, leadTime, varIn, varOut in tmp2:
    print("In:")
    for k,v in xIn.items():  
        print('_')
        print(f"{k.ljust(20)}: {v[:,:,0,0]}")
    print("Out:")
    for k,v in xOut.items():  
       print(f"{k.ljust(20)}: {v[:,0,0]}")
    break
#%%
class FakeData(IterableDataset):
    def __init__(self, dim):
        super().__init__()
        self._niter = 1
        self._nvar = 1
        self._vars = [f"k{i}" for i in range(self._nvar)]
        self.dim = dim
        
    def __iter__(self):
        for i in range(self._niter):
            yield {k: (i+1)*np.arange(self.dim) for k in self._vars}, {
                k: (i+1)*np.arange(self.dim) for k in self._vars
                }, self._vars, self._vars


def printNoLead(x):
    for xi, xo, vi, vo in x:
        print('In: ')     
    for k, v in xi.items():
        #print(f"{k.ljust(10)} --> {v}")
        print(f"{k} --> {v}")
    print('Out: ')
    for k,v in xo.items():
        #print(f"{k.ljust(10)} --> {v}")
        print(f"{k} --> {v}")
        
def printLead(x):
    for xi, xo, leadTime, vi, vo in x:
        print('In: ')
        print(f"lead time: {leadTime*100}")
        for k,v in xi.items():
            print(f"{k.ljust(10)} --> {v}")
        print('Out: ')
        for k,v in xo.items():
            print(f"{k.ljust(10)} --> {v}")

#%% usage
dataIn= FakeData(20)   
printNoLead(dataIn)   
dataDirect =  DirectForecast(dataIn, 'era5', pred_range=2, history=3, window=5)  
printNoLead(dataDirect)
## continuous
dataContinuousRand = ContinuousForecast(dataIn, random_lead_time=True, min_pred_range=1, max_pred_range=4, 
                                        hrs_each_step=1, history=3, window=5)
printLead(dataContinuousRand)
## Lead time fixed
dataContinuousFixed = ContinuousForecast(dataIn, random_lead_time=False, min_pred_range=6, max_pred_range=6, 
                                        hrs_each_step=1, history=4, window=3)
printLead(dataContinuousFixed)
#%% Individual Data Iter
class Downscale(IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for inp_data, out_data, variables, out_variables in self.dataset:
            inp_data = {
                k: torch.from_numpy(inp_data[k].astype(np.float32))
                for k in inp_data.keys()
            }
            out_data = {
                k: torch.from_numpy(out_data[k].astype(np.float32))
                for k in out_data.keys()
            }
            yield inp_data, out_data, variables, out_variables

class IndividualDataIter(IterableDataset):
    def __init__(
        self,
        dataset,
        transforms,
        output_transforms,
        subsample=6,
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.subsample = subsample

    def __iter__(self):
        for sample in self.dataset:
            if isinstance(self.dataset, (DirectForecast, Downscale)):
                inp, out, variables, out_variables = sample
            elif isinstance(self.dataset, ContinuousForecast):
                inp, out, lead_times, variables, out_variables = sample
            inp_shapes = set([inp[k].shape[0] for k in inp.keys()])
            out_shapes = set([out[k].shape[0] for k in out.keys()])
            assert len(inp_shapes) == 1
            assert len(out_shapes) == 1
            inp_len = next(iter(inp_shapes))
            out_len = next(iter(out_shapes))
            assert inp_len == out_len
            for i in range(0, inp_len, self.subsample):
                x = {k: inp[k][i] for k in inp.keys()}
                y = {k: out[k][i] for k in out.keys()}
                if self.transforms is not None:
                    if isinstance(self.dataset, (DirectForecast, ContinuousForecast)):
                        x = {
                            k: self.transforms[k](x[k].unsqueeze(1)).squeeze(1)
                            for k in x.keys()
                        }
                    elif isinstance(self.dataset, Downscale):
                        x = {
                            k: self.transforms[k](x[k].unsqueeze(0)).squeeze(0)
                            for k in x.keys()
                        }
                    else:
                        raise RuntimeError(f"Not supported task.")
                if self.output_transforms is not None:
                    y = {
                        k: self.output_transforms[k](y[k].unsqueeze(0)).squeeze(0)
                        for k in y.keys()
                    }
                if isinstance(self.dataset, (DirectForecast, Downscale)):
                    result = x, y, variables, out_variables
                elif isinstance(self.dataset, ContinuousForecast):
                    result = x, y, lead_times[i], variables, out_variables
                yield result
#%% Read the output file to see the data
## This code did not run
# Load the .npz file
# data = np.load(save_dir + '/test' + '/2017_0.npz')

# # Iterate over the keys of the data dictionary
# for key in data.keys():
#     print(f"The name of the array is '{key}'")

# tmp = data['2m_temperature']
# pr = data['total_precipitation']

# # ## check shape
# tmp.shape
# pr.shape

# # # Define dummy transformation functions using mean instead of lambda
# transforms = {
#     'input_var1': np.mean,
#     #'input_var2': np.mean,
# }

# output_transforms = {
#     'output_var1': np.mean,
#     #'output_var2': np.mean,
# }


# # Dummy dataset
# #dummy_dataset = [inp_data, out_data, ['input_var1', 'input_var2'], ['output_var1', 'output_var2']]
# dummy_dataset = [tmp, pr, ['2m_temperature'], ['total_precipitation']]
# # Create an instance of IndividualDataIter
# individual_data_iter = IndividualDataIter(dataContinuousRand, transforms=None, output_transforms=None, subsample=6)

# for sample in individual_data_iter:
#     # Unpack the sample
#     x, y = sample
#     print(sample)
#     # Print input and output data
#     print("Input data:")
#     for k, v in x.items():
#         print(f"{k}: {v}")
        
#     print("/nOutput data:")
#     for k, v in y.items():
#         print(f"{k}: {v}")
        
#     print("/nInput variables:", variables)
#     print("Output variables:", out_variables)
#     print("/n" + "="*50 + "/n")

#%% Using fake data
class FakeData(IterableDataset):
    def __init__(self, dim):
        super().__init__()
        self._niter = 1
        self._nvar = 2
        self._vars = [f"k{i}" for i in range(self._nvar)]
        self.dim = dim
        
    def __iter__(self):
        for i in range(self._niter):
            yield {k: (i+1)*np.arange(self.dim) for k in self._vars}, {
                k: (i+1)*np.arange(self.dim) for k in self._vars
                }, self._vars, self._vars

dataIn= FakeData(20)  
dataDirect =  DirectForecast(dataIn, 'era5', pred_range=2, history=3, window=5) 
individual_data_iter = IndividualDataIter(dataDirect, transforms=None, 
                                          output_transforms=None, subsample=2)

for sample in individual_data_iter:
       print(sample)   
#%% IterDataModule
# Standard library
import copy
import glob
import os
from typing import Dict, Optional

# Third party
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms
import pytorch_lightning as pl

# Local application
# from .iterdataset import (
#     NpyReader,
#     DirectForecast,
#     ContinuousForecast,
#     Downscale,
#     IndividualDataIter,
#     ShuffleIterableDataset,
# )


class IterDataModule(pl.LightningDataModule):
    """ClimateLearn's iter data module interface. Encapsulates dataset/task-specific
    data modules."""

    def __init__(
        self,
        task,
        inp_root_dir,
        out_root_dir,
        in_vars,
        out_vars,
        src=None,
        history=1,
        window=6,
        pred_range=6,
        random_lead_time=True,
        max_pred_range=120,
        hrs_each_step=1,
        subsample=1,
        buffer_size=10000,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        if task in ("direct-forecasting", "iterative-forecasting"):
            self.dataset_caller = DirectForecast
            self.dataset_arg = {
                "src": src,
                "pred_range": pred_range,
                "history": history,
                "window": window,
            }
            self.collate_fn = collate_fn
        elif task == "continuous-forecasting":
            self.dataset_caller = ContinuousForecast
            self.dataset_arg = {
                "random_lead_time": random_lead_time,
                "min_pred_range": pred_range,
                "max_pred_range": max_pred_range,
                "hrs_each_step": hrs_each_step,
                "history": history,
                "window": window,
            }
            self.collate_fn = collate_fn_continuous
        elif task == "downscaling":
            self.dataset_caller = Downscale
            self.dataset_arg = {}
            self.collate_fn = collate_fn

        self.inp_lister_train = sorted(
            glob.glob(os.path.join(inp_root_dir, "train", "*.npz"))
        )
        self.out_lister_train = sorted(
            glob.glob(os.path.join(out_root_dir, "train", "*.npz"))
        )
        self.inp_lister_val = sorted(
            glob.glob(os.path.join(inp_root_dir, "val", "*.npz"))
        )
        self.out_lister_val = sorted(
            glob.glob(os.path.join(out_root_dir, "val", "*.npz"))
        )
        self.inp_lister_test = sorted(
            glob.glob(os.path.join(inp_root_dir, "test", "*.npz"))
        )
        self.out_lister_test = sorted(
            glob.glob(os.path.join(out_root_dir, "test", "*.npz"))
        )

        self.transforms = self.get_normalize(inp_root_dir, in_vars)
        self.output_transforms = self.get_normalize(out_root_dir, out_vars)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.out_root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.out_root_dir, "lon.npy"))
        return lat, lon

    def get_data_variables(self):
        out_vars = copy.deepcopy(self.hparams.out_vars)
        if "2m_temperature_extreme_mask" in out_vars:
            out_vars.remove("2m_temperature_extreme_mask")
        return self.hparams.in_vars, out_vars

    def get_data_dims(self):
        lat = len(np.load(os.path.join(self.hparams.out_root_dir, "lat.npy")))
        lon = len(np.load(os.path.join(self.hparams.out_root_dir, "lon.npy")))
        forecasting_tasks = [
            "direct-forecasting",
            "iterative-forecasting",
            "continuous-forecasting",
        ]
        if self.hparams.task in forecasting_tasks:
            in_size = torch.Size(
                [
                    self.hparams.batch_size,
                    self.hparams.history,
                    len(self.hparams.in_vars),
                    lat,
                    lon,
                ]
            )
        elif self.hparams.task == "downscaling":
            in_size = torch.Size(
                [self.hparams.batch_size, len(self.hparams.in_vars), lat, lon]
            )
        ##TODO: change out size
        out_vars = copy.deepcopy(self.hparams.out_vars)
        if "2m_temperature_extreme_mask" in out_vars:
            out_vars.remove("2m_temperature_extreme_mask")
        out_size = torch.Size([self.hparams.batch_size, len(out_vars), lat, lon])
        return in_size, out_size

    def get_normalize(self, root_dir, variables):
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        return {
            var: transforms.Normalize(normalize_mean[var][0], normalize_std[var][0])
            for var in variables
        }

    def get_out_transforms(self):
        out_transforms = {}
        for key in self.output_transforms.keys():
            if key == "2m_temperature_extreme_mask":
                continue
            out_transforms[key] = self.output_transforms[key]
        return out_transforms

    def get_climatology(self, split="val"):
        path = os.path.join(self.hparams.out_root_dir, split, "climatology.npz")
        clim_dict = np.load(path)
        new_clim_dict = {}
        for var in self.hparams.out_vars:
            if var == "2m_temperature_extreme_mask":
                continue
            new_clim_dict[var] = torch.from_numpy(
                np.squeeze(clim_dict[var].astype(np.float32), axis=0)
            )
        return new_clim_dict

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if stage != "test":
            if not self.data_train and not self.data_val and not self.data_test:
                self.data_train = ShuffleIterableDataset(
                    IndividualDataIter(
                        self.dataset_caller(
                            NpyReader(
                                inp_file_list=self.inp_lister_train,
                                out_file_list=self.out_lister_train,
                                variables=self.hparams.in_vars,
                                out_variables=self.hparams.out_vars,
                                shuffle=True,
                            ),
                            **self.dataset_arg,
                        ),
                        transforms=self.transforms,
                        output_transforms=self.output_transforms,
                        subsample=self.hparams.subsample,
                    ),
                    buffer_size=self.hparams.buffer_size,
                )

                self.data_val = IndividualDataIter(
                    self.dataset_caller(
                        NpyReader(
                            inp_file_list=self.inp_lister_val,
                            out_file_list=self.out_lister_val,
                            variables=self.hparams.in_vars,
                            out_variables=self.hparams.out_vars,
                            shuffle=False,
                        ),
                        **self.dataset_arg,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                    subsample=self.hparams.subsample,
                )

                self.data_test = IndividualDataIter(
                    self.dataset_caller(
                        NpyReader(
                            inp_file_list=self.inp_lister_test,
                            out_file_list=self.out_lister_test,
                            variables=self.hparams.in_vars,
                            out_variables=self.hparams.out_vars,
                            shuffle=False,
                        ),
                        **self.dataset_arg,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                    subsample=self.hparams.subsample,
                )
        else:
            self.data_test = IndividualDataIter(
                self.dataset_caller(
                    NpyReader(
                        inp_file_list=self.inp_lister_test,
                        out_file_list=self.out_lister_test,
                        variables=self.hparams.in_vars,
                        out_variables=self.hparams.out_vars,
                        shuffle=False,
                    ),
                    **self.dataset_arg,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
                subsample=self.hparams.subsample,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )


def collate_fn(batch):
    def handle_dict_features(t: Dict[str, torch.tensor]) -> torch.tensor:
        t = torch.stack(tuple(t.values()))
        if len(t.size()) == 4:
            return torch.transpose(t, 0, 1)
        return t

    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))])
    has_extreme_mask = False
    for key in batch[0][1]:
        if key == "2m_temperature_extreme_mask":
            has_extreme_mask = True
    if not has_extreme_mask:
        out = torch.stack(
            [handle_dict_features(batch[i][1]) for i in range(len(batch))]
        )
        variables = list(batch[0][0].keys())
        out_variables = list(batch[0][1].keys())
        return inp, out, variables, out_variables
    out = []
    mask = []
    for i in range(len(batch)):
        out_dict = {}
        mask_dict = {}
        for key in batch[i][1]:
            if key == "2m_temperature_extreme_mask":
                mask_dict[key] = batch[i][1][key]
            else:
                out_dict[key] = batch[i][1][key]
        out.append(handle_dict_features(out_dict))
        if mask_dict != {}:
            mask.append(handle_dict_features(mask_dict))
    out = torch.stack(out)
    if mask != []:
        mask = torch.stack(mask)
    variables = list(batch[0][0].keys())
    out_variables = list(out_dict.keys())
    return inp, out, mask, variables, out_variables


def collate_fn_continuous(batch):
    def handle_dict_features(t: Dict[str, torch.tensor]) -> torch.tensor:
        t = torch.stack(tuple(t.values()))
        if len(t.size()) == 4:
            return torch.transpose(t, 0, 1)
        return t

    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))])
    out = torch.stack([handle_dict_features(batch[i][1]) for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    b, t, _, h, w = inp.shape
    lead_times = lead_times.reshape(b, 1, 1, 1, 1).repeat(1, t, 1, h, w)
    inp = torch.cat((inp, lead_times), dim=2)
    variables = list(batch[0][0].keys())
    out_variables = list(batch[0][1].keys())
    return inp, out, variables, out_variables

class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()
#%% Usage IterDataModule
# Define the task, input and output directories, input and output variables, etc.
inp_root_dir = 'C:/Users/Prakash/OneDrive - University of California, Davis/califo_work/seaonal_prediction/clim_learn_test/data/processed'
out_root_dir = inp_root_dir

# Initialize the IterDataModule
dm = IterDataModule(
    task="direct-forecasting",
    inp_root_dir=inp_root_dir,
    out_root_dir=out_root_dir,
    # in_vars=["2m_temperature", "geopotential"],
    # out_vars=["2m_temperature", "geopotential"],
    in_vars=["2m_temperature"],
    out_vars=["2m_temperature"],
   src="era5",
    subsample=12,
    pred_range=336,
    history=6,
    batch_size=32
)

# Setup the data module
dm.setup()

for inp, out, inVar, outVar in dm.test_dataloader():
    print(inp.shape, out.shape, inVar, outVar)

count = 0
for dt in dm.data_train:
    test = dt
    count += 1
    if count == 1:
        break
    
test[0]['2m_temperature'].shape
    

## See the input data
# Specify the path to your .npz file
#file_path = 'C:/Users/Prakash/OneDrive - University of California, Davis/califo_work/seaonal_prediction/downloaded_data_clim_learn/processed/test'
file_path = 'C:/Users/Prakash/OneDrive - University of California, Davis/califo_work/seaonal_prediction/clim_learn_test/data/processed/test'
# Load the data from the .npz file
data = np.load(inp_root_dir+'/test/2017_0.npz')
#data = np.load(file_path+'/climatology.npz')
#data = np.load(file_path+'/normalize_mean.npz')
data = np.load(file_path+'/normalize_std.npz')

# Iterate over the keys of the data dictionary
for k, v in data.items():
    print(f"Variable: {k}, Shape: {v.shape}")


# np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
# np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)


## Use dummy data
# Define the shape of the variables
shape = (6, 1, 32, 64) ## for all data
shape1 = (1, 32, 64) # for climatology
hh = np.array([1])
shape2 = hh.shape # normalized mean/std

# Create a dictionary to hold the variables
data1 = {
    "2m_temperature": np.ones(shape1),
    "total_cloud_cover": np.ones(shape1)
}

data1 = {
    "2m_temperature": np.ones(shape2),
    "total_cloud_cover": np.ones(shape2)
}

data2 = {
    "2m_temperature": np.zeros(shape2)+0.5,
    "total_cloud_cover": np.zeros(shape2)+0.5
}

path1 = 'C:/Users/Prakash/OneDrive - University of California, Davis/califo_work/seaonal_prediction/clim_learn_test/data/processed/test'
#path2 = 'C:/Users/Prakash/OneDrive - University of California, Davis/Desktop/test'
# Save the data to a single .npz file
#np.savez(path1+"/2017_0.npz", **data1)
np.savez(path1+"/climatology.npz", **data1)
np.savez(path1+"/normalize_std.npz", **data2)

 





























    
    
    
#%%

# Define your data loaders and other necessary components for training, validation, and testing

# Instantiate PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=2)

#for sample in individual_data_iter:
       # print(sample)


# Start training
trainer.fit(lit_module, train_dataloader, val_dataloader)

# Define DataLoader for the training dataset
## Need to load datasets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define DataLoader for the validation dataset
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


