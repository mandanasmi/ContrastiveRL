import os
import torch
import wandb
from absl import logging
from datetime import datetime
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gym 
# from moviepy.editor import ImageSequenceClip

def serialize_object(obj):
    return {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('__') and not callable(getattr(obj, attr))}

def set_global_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_global_log_levels(level):
    gym.logger.set_level(level)


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_n_params(model):
    return str(np.round(np.array([p.numel() for p in model.parameters()]).sum() / 1e6, 3)) + ' M params'

class Accumulator:
    def __init__(self):
        self.data = 0
        self.num_data = 0

    def reset_state(self):
        self.data = 0
        self.num_data = 0

    def update_state(self, tensor):
        with torch.no_grad():
            self.data += tensor
            self.num_data += 1

    def result(self):
        if self.num_data == 0:
            return None
        data = self.data.item() if hasattr(self.data, 'item') else self.data
        return float(data) / self.num_data

def get_idx_from_K(K, itself):
    """
    Returns the indexes of parents or children of itself for each instance in the batch
    """
    if K.shape[0] == 1:
        batch_idx = torch.nonzero(itself.unsqueeze(1) == K[0, :, 0])
        instance_idx = batch_idx[:, 0]
        K_idx = K[0, batch_idx[:, 1], 1]
    else:
        batch_idx = torch.nonzero(itself.unsqueeze(1) == K[..., 0])
        instance_idx = batch_idx[:, 0]
        K_idx = K[instance_idx, batch_idx[:, 1], 1]
    return instance_idx, K_idx


class DetectEMAThresh():
    """A class to detect plateaus of the loss with EMA."""
    def __init__(self, threshold=1., ema_decay=0.9, decay_steps=None, delta=0.1):
        self.ema_decay = ema_decay
        self.threshold = threshold
        self.ema = 0.
        self.decay_steps = decay_steps
        self.delta = delta
        self.n_steps = 0

    def __call__(self, loss):

        self.update_ema(loss)

        if self.ema < self.threshold:
            return True
        else:
            return False

    def update_ema(self, loss):
        self.ema = self.ema * self.ema_decay + loss * (1 - self.ema_decay)

    def scheduler_step(self):
        if self.decay_steps is not None:
            self.n_steps += 1
            if self.n_steps in self.decay_steps:
                self.threshold -= self.delta
        # else:
        #     raise AssertionError("decay_steps must be provided to use scheduler_step")

class DetectEMAPlateau():
    """A class to detect plateaus of the loss with EMA."""
    def __init__(self, patience=10, threshold=1e-4, ema_decay=0.9):
        self.ema_decay = ema_decay
        self.patience = patience
        self.threshold = threshold

        self.best = float('inf')

        self.epochs_counter = 0
        self.ema = 0.

    def __call__(self, loss):

        self.update_ema(loss)

        if self.ema < (self.best - self.threshold*self.best):
            self.epochs_counter = 0
            self.best = self.ema
            return False
        elif self.ema > (self.best + self.threshold*self.best):
            self.epochs_counter = 0
            self.best = self.ema
            return False
        else:
            self.epochs_counter += 1
            if self.epochs_counter >= self.patience:
                self.epochs_counter = 0
                self.best = self.ema
                return True
            else:
                return False

    def update_ema(self, loss):
        self.ema = self.ema * self.ema_decay + loss * (1 - self.ema_decay)

def get_NLL(gfn_B, X):
    """Fully observed setting"""
    with torch.no_grad():
        ll = gfn_B.logprobV(X.float(), "fixed", direction="B")

    return -ll

def get_NLL_det_encoder(QF, QB, X):
    with torch.no_grad():
        encoding = QF.Qnet(X)
        V_F = torch.cat([X, encoding], dim=1)
        ll = QB.probV(V_F, "full",
                      direction="B", include_H=True, log=True, reduction="sum")
    return -ll

def get_NLL_gibbs(V_repeat, QB, n_hiddens, batchsz_harmonic):
    logprob_QB = QB.probV(V_repeat, "full", direction="B", log=True, reduction="sum")
    logprob_QB = logprob_QB.view(-1, batchsz_harmonic)
    ll = n_hiddens * torch.log(torch.Tensor([2]).to(V_repeat.device)) - torch.log(torch.Tensor([1/batchsz_harmonic]).to(V_repeat.device)) - torch.logsumexp(-logprob_QB, dim=1)
    return -ll

def get_NLL_importance(gfn_F, gfn_B, X, batchsz_importance):

    batch_size = X.shape[0] * batchsz_importance
    X_repeat = torch.repeat_interleave(X.unsqueeze(1), batchsz_importance, dim=1).view(batch_size, -1)
    V_F = gfn_F.sampleV(batch_size, "full", direction="F", temp=1, epsilon=0,
                                X=X_repeat)
    #  logsumexp ( log QB(Hi,X) - log QF(Hi|X) ] - log K
    logprob_QB = gfn_B.probV(V_F, "full", direction="B", log=True, reduction="sum").view(-1, batchsz_importance)
    logprob_QF = gfn_F.probV(V_F, "full", direction="F", log=True, reduction="sum").view(-1, batchsz_importance)
    ll = torch.logsumexp(logprob_QB - logprob_QF, dim=1) - np.log(batchsz_importance)

    return -ll

def distance_nn_mnist(samp_gfn, test_set, ord=2):
    # Calculate L-ord distance with each test sample
    distances = torch.linalg.norm((test_set.float().view(len(test_set), -1).unsqueeze(0) - samp_gfn.unsqueeze(1)), dim=2, ord=ord)

    # Get distance with the nearest neighbor
    nn_dist = torch.min(distances, dim=1)[0]

    return nn_dist

