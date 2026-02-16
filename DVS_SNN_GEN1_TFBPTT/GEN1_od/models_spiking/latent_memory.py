import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Tuple, List, Optional, Dict
from torch import Tensor
from spikingjelly.activation_based import neuron, layer
from GEN1_od.models_spiking.neurons import MpLIFNode, Mp_ParametricLIFNode, LIFWOFireNode
from visualizer import get_local

class Latent_Memory(nn.Module):
    def __init__(self, cfg_latent_memory: Dict):
        super().__init__()
        self.H, self.W = cfg_latent_memory['latent_size']
        self.latent_dim = cfg_latent_memory['latent_dim']

        self.latent_neuron_type = cfg_latent_memory['latent_neuron_type']
        self.latent_tau = cfg_latent_memory['latent_tau']
        if self.latent_neuron_type == 'PLIF':
            self.latent_memory = neuron.ParametricLIFNode(init_tau=self.latent_tau, v_threshold=1., decay_input=False, step_mode='s', detach_reset=True)
        elif self.latent_neuron_type == 'PLIF_soft': # when spike, the the neuron's voltage will subtract `v_threshold`
            self.latent_memory = neuron.ParametricLIFNode(init_tau=self.latent_tau, v_threshold=1., decay_input=False, step_mode='s', detach_reset=True, v_reset=None)
        elif self.latent_neuron_type == 'LIF':
            self.latent_memory = neuron.LIFNode(tau=self.latent_tau, v_threshold=1., decay_input=False, step_mode='s', detach_reset=True)
        elif self.latent_neuron_type == 'LIF_soft':
            self.latent_memory = neuron.LIFNode(tau=self.latent_tau, v_threshold=1., decay_input=False, step_mode='s', detach_reset=True, v_reset=None)
        elif self.latent_neuron_type == 'Mp_PLIF':
            self.latent_memory = Mp_ParametricLIFNode(init_tau=self.latent_tau, v_threshold=1., decay_input=False, step_mode='s', detach_reset=True)
        elif self.latent_neuron_type == 'Mp_LIF':
            self.latent_memory = MpLIFNode(tau=self.latent_tau, v_threshold=1., decay_input=False, step_mode='s', detach_reset=True)
        elif self.latent_neuron_type == 'WOFire_LIF':
            self.latent_memory = LIFWOFireNode(tau=self.latent_tau, v_threshold=1., decay_input=False, step_mode='s', detach_reset=True)
        else:
            raise NotImplementedError('Latent Memory type not implemented.')

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim*2),
            nn.GELU(),
            nn.Linear(self.latent_dim*2, self.latent_dim),
        )
        self.norm_mlp = nn.LayerNorm(self.latent_dim)


        self.latent_mem_init_v = nn.Parameter(torch.normal(mean=0.05, std=0.01, size=(1, self.H * self.W, self.latent_dim)), requires_grad=True)

    def init_mem_states(self, batch_size):
        self.latent_memory.reset()
        self.latent_memory.v = self.latent_mem_init_v.repeat(batch_size, 1, 1)


    def forward(self, message: Tensor):
        if message is None:
            stimu = torch.zeros_like(self.latent_memory.v)

        else:
            lat = self.latent_memory.v.clone()
            z = lat + message
            stimu = self.mlp(self.norm_mlp(z))



        latent_spikes = self.latent_memory(stimu)





        return latent_spikes




