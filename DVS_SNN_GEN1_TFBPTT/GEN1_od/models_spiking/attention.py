import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Tuple, List, Optional, Dict
from torch import Tensor

from torch_scatter import scatter
from visualizer import get_local

class SparseOneWindowAttention_NoPadding(nn.Module):
    def __init__(self, cfg_attention=None):
        super().__init__()
        self.embed_out_dim = cfg_attention['embed_out_dim']
        self.latent_dim = cfg_attention['latent_dim']
        self.num_heads = cfg_attention['num_heads']
        self.out_dim = cfg_attention['out_dim']


        self.norm_latent = nn.LayerNorm(self.out_dim)
        self.kv = nn.Linear(self.embed_out_dim, self.latent_dim * self.num_heads *2, bias=True)
        self.q = nn.Linear(self.latent_dim * self.num_heads, self.latent_dim * self.num_heads, bias=True)
        self.scale = (self.latent_dim * self.num_heads) ** -0.5
        self.proj = nn.Linear(self.latent_dim * self.num_heads, self.out_dim, bias=True)
        self.proj_drop = nn.Dropout(0.5)




        self.noise_filter = False



    def softmax(self, weights, q_indices, z_shape, kv_shape):

        B, zL, _ = z_shape
        L, H, C = kv_shape


        max_weights = scatter(weights.detach(), q_indices, reduce='max', dim=0, dim_size=B*zL)
        if self.noise_filter:
            w_dust, q_indices_dust = self._prepair_dust(B, zL)
            max_weights = torch.maximum(max_weights, w_dust.detach())
            w_dust = (w_dust - max_weights).exp()
        weights = (weights - max_weights[q_indices]).exp()
        accm = scatter_add(weights, q_indices, dim=0, dim_size=B*zL, use_torch=True)
        if self.noise_filter:
            accm = accm + w_dust
        weights = weights / (accm[q_indices] + 1.0e-7)
        return weights




    def forward(self, latent_mem: Tensor, x: Tensor, q_indices: Tensor) -> Tensor:

        latent_mem = self.norm_latent(latent_mem)
        B, zL, C = latent_mem.shape


        kv = self.kv(x)
        kv = kv.view(-1, 2, self.num_heads, self.latent_dim).permute(1,0,2,3).contiguous()
        key, value = kv[0], kv[1]
        L, H, C = key.shape


        query = self.q(latent_mem).view(-1, self.num_heads, self.latent_dim)

        query = query * self.scale
        query = query[q_indices, :, :]
        weights = (key * query).sum(-1)

        weights = self.softmax(weights, q_indices, latent_mem.shape, key.shape)


        message = weights[:,:,None] * value
        message = message.view(L, -1)


        message = scatter_add(message, q_indices, dim=0, dim_size=B*zL, use_torch=True)

        message = message.view(B, zL, H*C)
        message = self.proj(message)
        message = self.proj_drop(message)

        return message


    def forward_fast_train(self, latent_mem: Tensor, key: Tensor, value: Tensor, q_indices):

        pass
        raise NotImplementedError




class DerormableAttention_NoPadding(nn.Module):
    def __init__(self, cfg_attention=None):
        super().__init__()
        self.embed_out_dim = cfg_attention['embed_out_dim']
        raise NotImplementedError

    def forward(self, latent_mem: Tensor, x: Tensor, q_indices: Tensor) -> Tensor:

        latent_mem = torch.ones((2, 60 * 76, 128))
        B, zL, C = latent_mem.shape

        return latent_mem



def scatter_add(values: Tensor, indices: Tensor, dim: int, dim_size: int, use_torch: bool = True, out: Optional[Tensor] = None) -> Tensor:
    if use_torch:
        return scatter(values, indices, reduce='add', dim=dim, dim_size=dim_size, out=out)
    else:
        if out is not None:
            output = out
        else:
            shape = list(values.shape)
            shape[dim] = dim_size
            output = torch.zeros(*shape, device=values.device, dtype=values.dtype)

        if indices.ndim != values.ndim:
            assert indices.ndim == 1
            view = [1] * values.ndim
            view[dim] = len(indices)
            repeat = list(values.shape)
            repeat[dim] = len(indices)
            indices = indices.view(*view).expand(*repeat)
        output.scatter_add_(dim, indices, values)
        return output
