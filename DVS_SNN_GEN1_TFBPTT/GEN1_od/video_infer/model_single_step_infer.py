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
        pass



    def forward_backbone(self, events, time_idx) -> Tensor:
        if time_idx == 0:
            self.latent_mem.init_mem_states(batch_size=1)


        latent_spikes = self.one_step(events, time_idx)


        latent_spikes = rearrange(latent_spikes, 'b (h w) c -> b c h w', h=self.latent_H, w=self.latent_W)


        output = self.detection.forward_backbone(latent_spikes)
        return output
    def forward_detect(self, output) -> Tensor:
        output = self.detection.forward_detect(output)
        return output




    def one_step(self, events, time_idx):
        embedding, q_indices = self.embedding(events, time_idx)
        if embedding is None:
            message = None
        else:
            latent_mem = self.latent_mem.latent_memory.v.clone()
            message = self.attention(latent_mem, embedding, q_indices)
        latent_spikes = self.latent_mem(message)
        return latent_spikes
