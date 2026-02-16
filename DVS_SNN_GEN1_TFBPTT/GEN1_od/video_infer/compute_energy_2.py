from thop import profile
from GEN1_od.models_spiking.builder import build_detection, build_embedding, build_attention, build_latent_mem
import torch.nn as nn
import torch
import numpy as np
from importlib import machinery
config_path = 'GEN1_od/config_test/config_yoloCS3_64attn_20e3_hard.py'
# config_path = 'GEN1_od/config_test/config_yolox_64attn_20e3_hard.py'
# config_path = 'GEN1_od/config_test/config_yoloCS3_128attn_20e3_hard.py'
config_module = machinery.SourceFileLoader('config', config_path).load_module()

cfg_embed = config_module.cfg_embed
cfg_attention = config_module.cfg_attention
cfg_latent_memory = config_module.cfg_latent_memory
cfg_Detection = config_module.cfg_Detection

class repn(nn.Module):
    def __init__(self, cfg_embed=None, cfg_attention=None, cfg_latent_memory=None):
        super().__init__()
        self.embedding = build_embedding(cfg_embed)
        self.attention = build_attention(cfg_attention)
        self.latent_mem = build_latent_mem(cfg_latent_memory)

    def forward(self, events_list):
        for time_idx, events in enumerate(events_list):
            if time_idx == 0:
                self.latent_mem.init_mem_states(batch_size=1)
                embedding, q_indices = self.embedding(events, time_idx)
            if embedding is None:
                message = None
            else:
                latent_mem = self.latent_mem.latent_memory.v.clone()
                message = self.attention(latent_mem, embedding, q_indices)
            latent_spikes = self.latent_mem(message)
        return latent_spikes


class snn_backbone(nn.Module):
    def __init__(self, cfg_detection=None):
        super().__init__()
        self.detection = build_detection(cfg_detection)
    def forward(self, events):
        output = self.detection.forward_backbone(events)
        return output

class snn_head(nn.Module):
    def __init__(self, cfg_detection=None):
        super().__init__()
        self.detection = build_detection(cfg_detection)
    def forward(self, output):
        output = self.detection.forward_detect(output)
        return output



N_events = 1835*4*5 # number of events in 100 ms
# N_events = 1835*4*5 / 5
ts = 5
fr= 0.1311
n_events_per_ts = int(N_events / ts)

repn_model = repn(cfg_embed, cfg_attention, cfg_latent_memory)
# input_repn = [torch.randn(1, n_events_per_ts,4 )] * ts
time = torch.randint(0, int(100e3/ts), (1, n_events_per_ts, 1))
x = torch.randint(0,304, (1, n_events_per_ts, 1))
y = torch.randint(0,240, (1, n_events_per_ts, 1))
p = torch.randint(0,2, (1, n_events_per_ts, 1))*2-1
input_repn = [torch.cat((time, x, y, p), dim=-1)] * ts
macs, params = profile(repn_model, inputs=(input_repn,))
print(f'repn_MAC: {macs/1e6}')
repn_eng = macs * 4.6 / 1e9





backbone_model = snn_backbone(cfg_Detection)
detect_model = snn_head(cfg_Detection)
input_backbone = torch.randn(1, 64, 60, 76)
macs, params = profile(backbone_model, inputs=(input_backbone,))
print(f'backbone_MAC: {macs/1e6}')
backbone_eng = macs *  fr * 0.9 / 1e9  * ts


input_detect = backbone_model(input_backbone)
macs, params = profile(detect_model, inputs=(input_detect,))
print(f'head_MAC: {macs/1e6}')
detect_eng = macs * 4.6 / 1e9


print(f'repn_eng: {repn_eng}')
print(f'backbone_eng: {backbone_eng}')
print(f'detect_eng: {detect_eng}')
print(f'total_eng: {repn_eng + backbone_eng + detect_eng}')


# macs, params = profile(self, inputs=(events,))
# eng = macs * 0.9 / 1e9 * 0.2404
