# https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py

import torch
from torch import nn
import time

from utils.common import LOGGER, colorstr
from utils.quantize import QuantLinear

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv3 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if  isinstance(v, QuantLinear):
            for p_name, p in v.named_parameters(recurse=0):
                g[3].append(p)  # weight (with decay)
        else:
            for p_name, p in v.named_parameters(recurse=0):
                if p_name == "bias":  # bias (no decay)
                    g[2].append(p)
                elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                    g[1].append(p)
                else:
                    g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    optimizer.add_param_group({"params": g[3], "weight_decay": 1e-4, "lr": 0.1, "momentum": 0.9})  # add g3 (QuantLinear weights)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias'
    )
    return optimizer

def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv3 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            elif p_name == "w": # tau (no decay)
                g[3].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    optimizer.add_param_group({"params": g[3], "weight_decay": 0.0})  # add g3 (tau weights)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias'
    )
    return optimizer