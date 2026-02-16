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
        # self.norm_input = nn.LayerNorm(self.embed_out_dim)
        self.kv = nn.Linear(self.embed_out_dim, self.latent_dim * self.num_heads *2, bias=True)
        self.q = nn.Linear(self.latent_dim * self.num_heads, self.latent_dim * self.num_heads, bias=True)
        self.scale = (self.latent_dim * self.num_heads) ** -0.5
        self.proj = nn.Linear(self.latent_dim * self.num_heads, self.out_dim, bias=True)
        self.proj_drop = nn.Dropout(0.5)




        self.noise_filter = False



    def softmax(self, weights, q_indices, z_shape, kv_shape):

        B, zL, _ = z_shape
        L, H, C = kv_shape


        max_weights = scatter(weights.detach(), q_indices, reduce='max', dim=0, dim_size=B*zL)    # (B*zL, H)
        if self.noise_filter:
            w_dust, q_indices_dust = self._prepair_dust(B, zL)
            max_weights = torch.maximum(max_weights, w_dust.detach()) # here just use the positive weight, because the w_dust are all 0
            w_dust = (w_dust - max_weights).exp()
        weights = (weights - max_weights[q_indices]).exp()                               # (L, H)
        accm = scatter_add(weights, q_indices, dim=0, dim_size=B*zL, use_torch=True)    # (B*zL, H)
        if self.noise_filter:
            accm = accm + w_dust
        weights = weights / (accm[q_indices] + 1.0e-7)
        return weights



    @get_local('message_attention')
    def forward(self, latent_mem: Tensor, x: Tensor, q_indices: Tensor) -> Tensor:
        # input latent_mem shape: (B , zL, latent_dim * num_heads), here we use the self.latent_memory.v as the latent memory
        # input x shape: (N, embed_out_dim)

        # latent_mem = torch.ones((2, 60 * 76, 128))
        latent_mem = self.norm_latent(latent_mem)
        B, zL, C = latent_mem.shape  # (B, zL, latent_dim * num_heads), zL=60*76
        # print('------------------------------')


        # get kv
        kv = self.kv(x)  # (N, 2*H*D)
        kv = kv.view(-1, 2, self.num_heads, self.latent_dim).permute(1,0,2,3).contiguous()  # (2, N, H, D)
        key, value = kv[0], kv[1] # (N, H, D), (N, H, D)
        L, H, C = key.shape

        # get query from the latents
        query = self.q(latent_mem).view(-1, self.num_heads, self.latent_dim) # (B*zL, H, D)


        query = query * self.scale

        query = query[q_indices, :, :]  # (N, H, D)

        weights = (key * query).sum(-1)    # shape: [N,4,32]-> [N,4]

        weights = self.softmax(weights, q_indices, latent_mem.shape, key.shape) # 这里softmax或者说attention,本质上是用来计算同一个cell中，哪些events有更大的权重

        message = weights[:,:,None] * value      #weights: (N, 4), value:(N,4,32), message:(N,4,32)

        message = message.view(L, -1)  # (N, HC)


        # aggregate
        message = scatter_add(message, q_indices, dim=0, dim_size=B*zL, use_torch=True) # (B*zL, HC)

        message_attention = message.clone()
        # projection
        message = message.view(B, zL, H*C) # (B, zL, HC)

        message = self.proj(message) # (B, zL, out_dim)


        message = self.proj_drop(message)


        return message


    def forward_fast_train(self, latent_mem: Tensor, key: Tensor, value: Tensor, q_indices):


        pass
        raise NotImplementedError






class DerormableAttention_NoPadding(nn.Module):
    def __init__(self, cfg_attention=None):
        super().__init__()
        self.embed_out_dim = cfg_attention['embed_out_dim']
        # self.latent_size = cfg_attention['latent_size']
        raise NotImplementedError

    def forward(self, latent_mem: Tensor, x: Tensor, q_indices: Tensor) -> Tensor:
        # input latent_mem shape: (B , zL, latent_dim * num_heads)
        # input x shape: (N, embed_out_dim)

        latent_mem = torch.ones((2, 60 * 76, 128))
        B, zL, C = latent_mem.shape  # (B, zL, latent_dim * num_heads), zL=60*76

        return latent_mem



def scatter_add(values: Tensor, indices: Tensor, dim: int, dim_size: int, use_torch: bool = True, out: Optional[Tensor] = None) -> Tensor:
    if use_torch:
        return scatter(values, indices, reduce='add', dim=dim, dim_size=dim_size, out=out)
    else:
        if out is not None:
            output = out
        else:
            shape = list(values.shape)
            shape[dim] = dim_size # shape = [13680, 4]
            output = torch.zeros(*shape, device=values.device, dtype=values.dtype) # output:(13680,4)

        if indices.ndim != values.ndim: # indices: (10828,), values: (10828, 4)
            # broadcast
            assert indices.ndim == 1
            view = [1] * values.ndim # [1] * 2 = [1, 1]
            view[dim] = len(indices) # view[0] = 10828, view = [10828, 1]
            repeat = list(values.shape) # [10828, 4]
            repeat[dim] = len(indices) # repeat[0] = 10828, repeat = [10828, 4]
            indices = indices.view(*view).expand(*repeat) # indices: (10828, 4)
        output.scatter_add_(dim, indices, values)
        return output