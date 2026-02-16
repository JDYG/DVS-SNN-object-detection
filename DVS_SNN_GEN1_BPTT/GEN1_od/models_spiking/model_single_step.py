import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from GEN1_od.models_spiking.builder import build_embedding, build_attention, build_latent_mem, build_detection

import time



class Gen1Spiking(nn.Module):
    def __init__(self, cfg_embed=None, cfg_attention=None, cfg_latent_memory=None, cfg_detection=None, cfg_pretrain=None, exec_string=None):
        super().__init__()

        self.latent_H = cfg_latent_memory['latent_size'][0]
        self.latent_W = cfg_latent_memory['latent_size'][1]

        self.embedding = build_embedding(cfg_embed)
        self.attention = build_attention(cfg_attention)
        self.latent_mem = build_latent_mem(cfg_latent_memory)
        self.detection = build_detection(cfg_detection)
        self.init_weights(cfg_pretrain, exec_string)


    def init_weights(self, cfg_pretrain, exec_string):


        exec(exec_string)











    def forward(self, events_list: list) -> Tensor:

        batch_size = len(events_list[0])
        time_steps = len(events_list)

        fast_train = False
        if fast_train:
            # extract key, value and query_indices in advance for fast trainig
            events_list = self.fast_train_kv(events_list)
        for time_idx, events in enumerate(events_list):

            if time_idx == 0:
                self.latent_mem.init_mem_states(batch_size)
            if fast_train:
                latent_spikes = self.fast_train_one_step(events)
            else:
                latent_spikes = self.one_step(events, time_idx) # latent_spikes (B, 60*76, latent_dim)



            latent_spikes = rearrange(latent_spikes, 'b (h w) c -> b c h w', h=self.latent_H, w=self.latent_W)


            output = self.detection(latent_spikes, profile=False, last_step = time_idx==time_steps-1) # output [(bs,3,60,76,7), (bs,3,30,38,7), (bs,3,15,19,7)]

        return output


    def fast_train_kv(self,events_list):
        H = self.attention.num_heads
        C = self.attention.latent_dim
        ev_tensor, ev_q_indices, split_sizes = self.embedding.forward_fast_train(events_list)  # embedding
        if ev_tensor is None:
            events_list = [[list(), list(), list()] for _ in range(len(events_list))]
        else:
            kv = self.attention.kv(ev_tensor)  # ev_tensor(N,96), kv(N,2*H*C)
            kv = kv.view(-1, 2, H, C).permute(1, 0, 2, 3).contiguous()  # (2, N, H, C)
            key, value = kv[0], kv[1]  # (N, H, C), (N, H, C)
            list_keys = key.split(split_sizes)
            list_values = value.split(split_sizes)
            list_ev_q_indices = ev_q_indices.split(split_sizes)
            events_list = list(zip(list_keys, list_values, list_ev_q_indices))  # key, value are pre-computed for attention
        return events_list

    def fast_train_one_step(self, events): # fast training to generate one step spikes from latent memory
        key, value, query_indices = events
        if len(key) == 0:
            message = None
        else:
            latent_mem = self.latent_mem.latent_memory.v.clone()
            message = self.attention.forward_fast_train(latent_mem, key, value, query_indices)
        latent_spikes = self.latent_mem(message)
        return latent_spikes

    def one_step(self, events, time_idx): # generate one step spikes from latent memory

        embedding, q_indices = self.embedding(events, time_idx)
        if embedding is None:
            message = None
        else:
            latent_mem = self.latent_mem.latent_memory.v.clone()
            message = self.attention(latent_mem, embedding, q_indices)
        latent_spikes = self.latent_mem(message)
        return latent_spikes
