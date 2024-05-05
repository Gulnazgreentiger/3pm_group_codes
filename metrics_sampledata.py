# Standard library
from dataclasses import dataclass
from functools import wraps
from typing import List, Union, Optional

# Third party
import torch.nn.functional as F
import numpy.typing as npt
import torch
import torch.distributions as dist
from typing import Tuple

Pred = Union[torch.FloatTensor, torch.DoubleTensor, torch.distributions.Normal]


@dataclass
class MetricsMetaInfo:
    in_vars: List[str]
    out_vars: List[str]
    lat: npt.ArrayLike
    lon: npt.ArrayLike
    climatology: torch.Tensor


METRICS_REGISTRY = {}


def register(name):
    def decorator(metric_class):
        METRICS_REGISTRY[name] = metric_class
        metric_class.name = name
        return metric_class

    return decorator


def handles_probabilistic(metric):
    @wraps(metric)
    def wrapper(pred: Pred, *args, **kwargs):
        if isinstance(pred, torch.distributions.Normal):
            pred = pred.loc
        return metric(pred, *args, **kwargs)

    return wrapper

#%%

def generate_sample_data(batch_size: int, num_channels: int, image_size: Tuple[int, int]) -> Tuple[torch.distributions.Normal, torch.Tensor]:
    mean = torch.randn(batch_size, num_channels, *image_size)
    std = torch.rand(batch_size, num_channels, *image_size) * 0.5 + 0.1  # Ensure std is positive and not too small
    pred_distribution = dist.Normal(mean, std)

    target = torch.randn(batch_size, num_channels, *image_size)  # Random target values

    return pred_distribution, target

def gaussian_crps(
    pred: torch.distributions.Normal,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    mean, std = pred.loc, pred.scale
    z = (target - mean) / std
    standard_normal = torch.distributions.Normal(
        torch.zeros_like(mean), torch.ones_like(mean)
    )
    pdf = torch.exp(standard_normal.log_prob(z))
    cdf = standard_normal.cdf(z)
    crps = std * (z * (2 * cdf - 1) + 2 * pdf - 1 / torch.pi)
    if lat_weights is not None:
        crps = crps * lat_weights
    per_channel_losses = crps.mean([0, 2, 3])
    loss = crps.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))

# Test the function
batch_size = 8
num_channels = 3
image_size = (32, 32)

pred_distribution, target = generate_sample_data(batch_size, num_channels, image_size)
pred_mean = pred_distribution.mean  # Extract mean from the distribution
loss = gaussian_crps(pred_distribution, target)

print("Loss shape:", loss.shape)
print("Loss values:")
print(loss)



#%%
def generate_sample_data_spread(batch_size: int, num_channels: int, image_size: Tuple[int, int]) -> torch.distributions.Normal:
    mean = torch.randn(batch_size, num_channels, *image_size)
    std = torch.rand(batch_size, num_channels, *image_size) * 0.5 + 0.1  # Ensure std is positive and not too small
    pred_distribution = dist.Normal(mean, std)

    return pred_distribution

def gaussian_spread(
    pred: torch.distributions.Normal,
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    variance = torch.square(pred.scale)
    if lat_weights is not None:
        variance = variance * lat_weights
    per_channel_losses = variance.mean([2, 3]).sqrt().mean(0)
    loss = variance.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


# Test the function
batch_size = 8
num_channels = 3
image_size = (32, 32)

pred_distribution_spread = generate_sample_data_spread(batch_size, num_channels, image_size)
loss_spread = gaussian_spread(pred_distribution_spread)

print("Loss shape:", loss_spread.shape)
print("Loss values:")
print(loss_spread)

#%%

def generate_sample_data_nrmses(
    batch_size: int, num_channels: int, image_size: Tuple[int, int], clim_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = torch.randn(batch_size, num_channels, *image_size)  # Random predictions
    target = torch.randn(batch_size, num_channels, *image_size)  # Random targets
    clim = torch.randn(num_channels, *clim_size)  # Random climatology data

    return pred, target, clim

def nrmses(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    clim: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    y_normalization = clim.squeeze()
    error = (pred.mean(dim=0) - target.mean(dim=0)) ** 2  # (C, H, W)
    if lat_weights is not None:
        error = error * lat_weights.squeeze(0)
    per_channel_losses = error.mean(dim=(-2, -1)).sqrt() / y_normalization.mean(dim=(-2, -1))  # C
    loss = per_channel_losses.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))



# Test the function
batch_size = 8
num_channels = 3
image_size = (32, 32)
clim_size = (32, 32)

pred, target, clim = generate_sample_data_nrmses(batch_size, num_channels, image_size, clim_size)
# clim = torch.from_numpy([])
loss_nrmses = nrmses(pred, target, clim)

print("Loss shape:", loss_nrmses.shape)
print("Loss values:")
print(loss_nrmses)
