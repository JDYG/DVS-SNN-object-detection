import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from torch import Tensor
import math
from einops import rearrange


class EventEmbedding_NoPadding(nn.Module):
    def __init__(self, cfg_embed: Dict):
        super().__init__()

        self.shift_normalize = cfg_embed['shift_normalize']
        self.scale_normalize = cfg_embed['scale_normalize']
        self.H = cfg_embed['input_size'][0]
        self.W = cfg_embed['input_size'][1]
        self.duration = cfg_embed['duration']
        self.time_delta = cfg_embed['duration'] / cfg_embed['time_bins']
        self.latent_h = cfg_embed['latent_size'][0]
        self.latent_w = cfg_embed['latent_size'][1]
        self.window_h = self.H / self.latent_h # 240 / 60 = 4
        self.window_w = self.W / self.latent_w # 304 / 76 = 4
        self.xy_relative = cfg_embed['xy_relative']
        self.discrete_time = cfg_embed['discrete_time']

        # region embddding_type
        if cfg_embed['embed_type'][0] == 'MLP':
            self.embedding_xy = Embedding_MLP(input_dim=2, dynamic_dim=cfg_embed['dynamic_dim'][0],
                                              out_dim=cfg_embed['out_dim'][0], activation=cfg_embed['activation'])
        elif cfg_embed['embed_type'][0] == 'Positional':
            self.embedding_xy = PositionalFourier(input_dim=2, dynamic_dim=cfg_embed['dynamic_dim'][0],
                                                  out_dim=cfg_embed['out_dim'][0],
                                                  max_log_scale=cfg_embed['max_log_scale'][0])
        elif cfg_embed['embed_type'][0] == 'Gaussian':
            self.embedding_xy = GaussianFourier(input_dim=2, dynamic_dim=cfg_embed['dynamic_dim'][0],
                                                out_dim=cfg_embed['out_dim'][0])
        else:
            raise NotImplementedError

        if cfg_embed['embed_type'][1] == 'MLP':
            self.embedding_t = Embedding_MLP(input_dim=1, dynamic_dim=cfg_embed['dynamic_dim'][1],
                                             out_dim=cfg_embed['out_dim'][1], activation=cfg_embed['activation'])
        elif cfg_embed['embed_type'][1] == 'Positional':
            self.embedding_t = PositionalFourier(input_dim=1, dynamic_dim=cfg_embed['dynamic_dim'][1],
                                                 out_dim=cfg_embed['out_dim'][1],
                                                 max_log_scale=cfg_embed['max_log_scale'][1])
        elif cfg_embed['embed_type'][1] == 'Gaussian':
            self.embedding_t = GaussianFourier(input_dim=1, dynamic_dim=cfg_embed['dynamic_dim'][1],
                                               out_dim=cfg_embed['out_dim'][1])
        else:
            raise NotImplementedError

        if cfg_embed['embed_type'][2] == 'MLP':
            self.embedding_p = Embedding_MLP(input_dim=1, dynamic_dim=cfg_embed['dynamic_dim'][2],
                                             out_dim=cfg_embed['out_dim'][2], activation=cfg_embed['activation'])
        elif cfg_embed['embed_type'][2] == 'Positional':
            self.embedding_p = PositionalFourier(input_dim=1, dynamic_dim=cfg_embed['dynamic_dim'][2],
                                                 out_dim=cfg_embed['out_dim'][2],
                                                 max_log_scale=cfg_embed['max_log_scale'][2])
        elif cfg_embed['embed_type'][2] == 'Gaussian':
            self.embedding_p = GaussianFourier(input_dim=1, dynamic_dim=cfg_embed['dynamic_dim'][2],
                                               out_dim=cfg_embed['out_dim'][2])
        else:
            raise NotImplementedError
        # endregion

        # self.norm = nn.GroupNorm(3, cfg_embed['out_dim'][0] + cfg_embed['out_dim'][1] + cfg_embed['out_dim'][2])
        # self.norm = nn.BatchNorm1d(cfg_embed['out_dim'][0] + cfg_embed['out_dim'][1] + cfg_embed['out_dim'][2])
        self.norm = nn.LayerNorm(cfg_embed['out_dim'][0] + cfg_embed['out_dim'][1] + cfg_embed['out_dim'][2])
    def preproc_events(self, events, time_idx): # events is a list of tensor
        output = []
        for bidx in range(len(events)):
            evt = events[bidx]
            if len(evt) == 0:
                output.append(torch.empty(0,5, dtype=evt.dtype, device=evt.device))
                continue

            t0 = time_idx * self.duration
            evt[:,0] = evt[:,0] - t0
            b = torch.tensor([bidx] * len(evt), dtype=evt.dtype, device=evt.device).view(-1,1)
            out = torch.cat([evt, b], dim=1)
            output.append(out)
        output = torch.cat(output, dim=0)

        return output[:,0], output[:,1], output[:,2], output[:,3], output[:,4]


    def forward_fast_train(self, events_list: List):
        split_sizes = []
        batch_indices = []
        evdata = []
        for time_index, events in enumerate(events_list):
            dt, x, y, p, b = self.preproc_events(events, time_index)
            ev = torch.stack([dt, x, y, p], dim=-1)
            evdata.append(ev)
            batch_indices.append(b)
            split_sizes.append(len(dt))
        evdata = torch.cat(evdata, dim=0) # (N,4) contact all events in all batch and time bins
        batch_indices = torch.cat(batch_indices, dim=0)
        dt, x, y, p = evdata[:, 0], evdata[:, 1], evdata[:, 2], evdata[:, 3]
        ev_tensor, ev_q_indices = self._forward(dt, x, y, p, batch_indices) # ev_tensor shape:(N, 3*32)

        return ev_tensor, ev_q_indices, split_sizes



    def forward(self, events: List, time_idx) -> Tensor:
        # input: events 里面是batch size个list，每个list里面是一个tensor(Nx4)，N是events的数目, p:-1/1
        dt, x, y, p, b = self.preproc_events(events, time_idx) # output p is [-1,1]
        if len(dt) == 0:
            return None, None
        # window indices
        wx = torch.div(x, self.window_w, rounding_mode='trunc')
        wy = torch.div(y, self.window_h, rounding_mode='trunc')
        # indices = (b * self.latent_h * self.latent_w + wy * self.latent_w + wx).long()
        embedding, indices = self._forward(dt, x, y, p, b)
        return embedding, indices
    def _forward(self, dt, x, y, p, b) -> Tensor:
        # input: events 里面是batch size个list，每个list里面是一个tensor(Nx4)，N是events的数目, p:-1/1
        # if len(events) >= 1: # batch size > 1
        #     dt, x, y, p, b = self.preproc_events(events, time_idx)
        #     if len(dt) == 0:
        #         return None, None
        #     # window indices
        #     wx = torch.div(x, self.window_w, rounding_mode='trunc')
        #     wy = torch.div(y, self.window_h, rounding_mode='trunc')
        #     indices = (b * self.latent_h * self.latent_w + wy * self.latent_w + wx).long()

        # else:  # batch size = 1
        #     events = events[0]
        #     if len(events) == 0:
        #         return None, None
        #     t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        #     t0 = time_idx * self.duration
        #     dt = t - t0
        #     wx = torch.div(x, self.window_w, rounding_mode='trunc')
        #     wy = torch.div(y, self.window_h, rounding_mode='trunc')
        #     indices = (wy * self.latent_w + wx).long()


        wx = torch.div(x, self.window_w, rounding_mode='trunc')
        wy = torch.div(y, self.window_h, rounding_mode='trunc')
        indices = (b * self.latent_h * self.latent_w + wy * self.latent_w + wx).long()

        if self.xy_relative:
            x = x % self.window_w
            y = y % self.window_h
            xy = torch.stack([x, y], dim=-1)
            if self.shift_normalize[0]:
                xy = xy - torch.tensor([self.window_w, self.window_h], device=xy.device).float() * 0.5
            if self.scale_normalize[0]:
                xy = xy / torch.tensor([self.window_w, self.window_h], device=xy.device).float()
        else:
            xy = torch.stack([x, y], dim=-1)
            if self.shift_normalize[0]:
                xy = xy - torch.tensor([self.W, self.H], device=xy.device).float() * 0.5
            if self.scale_normalize[0]:
                xy = xy / torch.tensor([self.W, self.H], device=xy.device).float()

        if self.discrete_time:
            dt = torch.div(dt, self.time_delta, rounding_mode='trunc')
        else:
            dt = dt / self.time_delta
        if self.shift_normalize[1]:  # t , False
            dt = dt - self.duration * 0.5
        if self.scale_normalize[1]: # t, False
            dt = dt / self.duration

        if self.scale_normalize[2]: # p, wheather to convert the polarity to [0, 1]
            p = (p + 1)/2


        embedding_xy = self.embedding_xy(xy)
        embedding_t = self.embedding_t(dt.unsqueeze(-1))
        embedding_p = self.embedding_p(p.unsqueeze(-1))
        embedding = torch.cat([embedding_xy, embedding_t, embedding_p], dim=-1) # (N, 3*out_dim) (N,96)
        embedding = self.norm(embedding) # (N, 3*out_dim) (N,96) !!!!!!!!!
        return embedding, indices # embedding: (N, 3*out_dim) (N,96), indices: (N,)












class Embedding_MLP(nn.Module):
    def __init__(self, input_dim: int, dynamic_dim: int, out_dim: int, activation = 'relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, dynamic_dim)
        self.norm1 = nn.LayerNorm(dynamic_dim) # only norm the dims of each one event
        self.activation = get_activation(activation)
        self.fc2 = nn.Linear(dynamic_dim, out_dim)
        # self.norm2 = nn.LayerNorm(out_dim)
        # self.activation2 = get_activation(activation)


    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # x = self.norm2(x)
        return x


# https://github.com/matajoh/fourier_feature_nets/tree/main
class Embedding_FourierFeature(nn.Module):
    def __init__(self, input_dim: int, dynamic_dim: int, out_dim: int, a_values: torch.Tensor, b_values: torch.Tensor):
        super().__init__()

        assert b_values.shape[0] == input_dim
        assert a_values.shape[0] == b_values.shape[1]
        assert b_values.shape[1] * 2 == dynamic_dim # dynamic dim 是sin / cos stack后的维度， b_values.shape[1] 是Fourier Feature对单独sin/cos的输出维度
        self.a_values = nn.Parameter(a_values, requires_grad=False)
        self.b_values = nn.Parameter(b_values, requires_grad=False)

        self.fc1 = nn.Linear(dynamic_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        x = x.float() # when no padding, input x shape (N,2), input x range should be [0, 1]
        encoded = (2*math.pi * x) @ self.b_values
        encoded = torch.cat([self.a_values * torch.cos(encoded), self.a_values * torch.sin(encoded)], dim=-1)
        encoded = self.fc1(encoded)
        encoded = self.norm1(encoded)
        return encoded

class PositionalFourier(Embedding_FourierFeature):
    def __init__(self, input_dim: int, dynamic_dim: int, out_dim: int, max_log_scale: float):
       
        embedding_size = dynamic_dim / 2
        b_values = self._encoding(max_log_scale, embedding_size, input_dim)
        a_values = torch.ones(b_values.shape[1])
        Embedding_FourierFeature.__init__(self, input_dim=input_dim, out_dim=out_dim, dynamic_dim=dynamic_dim, a_values=a_values, b_values=b_values)
    
    @staticmethod
    def _encoding(max_log_scale: float, embedding_size: int, input_dim: int):
        """Produces the encoding b_values matrix."""
        embedding_size = int(embedding_size // input_dim)
        frequencies_matrix = 2. ** torch.linspace(0, max_log_scale, embedding_size)
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(input_dim) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, input_dim)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        return frequencies_matrix


class GaussianFourier(Embedding_FourierFeature):
    def __init__(self, input_dim: int, dynamic_dim: int, out_dim: int, sigma: float=1.0):
        embedding_size = int(dynamic_dim / 2)
        b_values = torch.normal(0, sigma, size=(input_dim, embedding_size))
        a_values = torch.ones(b_values.shape[1])
        Embedding_FourierFeature.__init__(self, input_dim=input_dim, out_dim=out_dim, dynamic_dim=dynamic_dim, a_values=a_values, b_values=b_values)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "none":
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module
