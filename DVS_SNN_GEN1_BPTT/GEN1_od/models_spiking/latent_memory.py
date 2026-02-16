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
            # nn.Dropout(0.0),
            nn.Linear(self.latent_dim*2, self.latent_dim),
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.latent_dim),
        #     # nn.GELU(),
        #     # nn.Dropout(0.0),
        #     # nn.Linear(self.latent_dim*2, self.latent_dim),
        # )

        self.norm_mlp = nn.LayerNorm(self.latent_dim)


        self.latent_mem_init_v = nn.Parameter(torch.normal(mean=0.001, std=0.001, size=(1, self.H * self.W, self.latent_dim)), requires_grad=True)

    def init_mem_states(self, batch_size):
        self.latent_memory.reset()
        self.latent_memory.v = self.latent_mem_init_v.repeat(batch_size, 1, 1)

    # @get_local('stimu')
    @get_local('latent_mem_charge')
    def forward(self, message: Tensor):
        # input message shape: (B, zL, latent_dim) zL = 60*76
        # self.latent_memory.v shape (B, zL, latent_dim)
        if message is None:
            stimu = torch.zeros_like(self.latent_memory.v)
            return self.latent_memory(stimu)

        lat = self.latent_memory.v.clone()
        z = lat + message
        # z = z + self.mlp(self.norm_mlp(z))


        #######################
        # 2025-07-11
        # gaoshy: test if add the z to sitmu has correct result
        # stimu = self.mlp(self.norm_mlp(z)) + z

        # stimu = self.mlp(self.norm_mlp(z)) + message
        #######################


        stimu = self.mlp(self.norm_mlp(z)) # original transformer z = MLP(LayerNorm(z'))+z', ref: Vision transformer with deformable attention
        # because we use the membrane potential the latent memory, after the neuron, the stimulus will add to membrane potential acutomatically
        # therefore, here we don't add z again

        ## need to recover (uncomment)
        latent_spikes = self.latent_memory(stimu)
        ####

        #
        # ##---just used to acquire the latent memory before the firing for visualization
        # self.latent_memory.v_float_to_tensor(stimu)
        # self.latent_memory.neuronal_charge(stimu)
        # latent_mem_charge = self.latent_memory.v.clone()
        # latent_spikes = self.latent_memory.neuronal_fire()
        # self.latent_memory.neuronal_reset(latent_spikes)
        #
        # ##---


        return latent_spikes




