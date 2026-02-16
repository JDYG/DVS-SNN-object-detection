import torch
import torch.nn as nn
from torch import Tensor
import contextlib
import math
from typing import Dict, List, Tuple
from spikingjelly.activation_based import layer
from GEN1_od.models_spiking.blocks import *

from utils.common import LOGGER
from utils.torch_utils import time_sync


try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

class Detect_Yolo(nn.Module):
    strides = None # strides computed during build
    dynamic = False # force grid reconstruction
    export = False # export mode

    def __init__(self, nc=2, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor, 2+5=7
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors, 3
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid, [tensor([]), tensor([]), tensor([])]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(layer.Conv2d(x, self.no * self.na, 1) for x in ch) # output conv
        self.inplace = inplace  # use in-place activation




    def forward(self, x):
        # input x is [(bs, 128, 60, 76), (bs, 256, 30, 38), (bs, 512, 15, 19)]
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i]) # conv
            bs, _, ny, nx = x[i].shape  # x(bs,21,60,76) to x(bs,3,60,76,7)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:  # inference, when inference, output is xywh
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                 # Detect (boxes only)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.strides[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        if self.training:
            return x #x [(bs,3,80,80,85), (bs,3,40,40,85), (bs,3,20,20,85)]
        else:
            return (torch.cat(z, 1), x)  # the x is returned for compute loss

            # return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
            # x [(bs,3,80,80,85), (bs,3,40,40,85), (bs,3,20,20,85)]

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.strides[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Detect_Yolox(nn.Module):
    def __init__(self, num_classes, feat_channels=256, stacked_convs=2,in_channels=[256, 256, 256], use_stem=True):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs

        self.levels = nn.ModuleList()
        for i in range(len(in_channels)):
            self.levels.append(
                BBoxHead(
                    num_classes, self.n_anchors, in_channels[i], feat_channels,
                    act="silu", depthwise=False, use_stem=use_stem, stacked_convs=stacked_convs))
        print('Use Detect_Yolox')

    def forward(self, xin):
        # input x is [(bs, 128, 60, 76), (bs, 256, 30, 38), (bs, 512, 15, 19)]
        x1, x2, x3 = xin
        # print(f'x1.mena:{x1.mean()}, x2.mean:{x2.mean()}, x3.mean:{x3.mean()}')
        reg1, obj1, cls1 = self.levels[0](x1)  # reg1:(B,4,60,76), obj1: (B,1,60,76), cls1: (B,2,60,76)
        reg2, obj2, cls2 = self.levels[1](x2)
        reg3, obj3, cls3 = self.levels[2](x3)

        if self.training:
            return (reg1, reg2, reg3), (obj1, obj2, obj3), (cls1, cls2, cls3)
        else:  # inference
            reg_outputs, obj_outputs, cls_outputs = (reg1, reg2, reg3), (obj1, obj2, obj3), (cls1, cls2, cls3)

            outputs = []
            for k in range(len(reg_outputs)):
                reg_output = reg_outputs[k]  # the k layer of the pyramid feature
                obj_output = obj_outputs[k]
                cls_output = cls_outputs[k]
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                outputs.append(output)

            hw = [(x.shape[-2], x.shape[-1]) for x in
                  outputs]  # the height and width of the feature map, [(60,76),(30,38),(15,19)]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2,
                                                                                          1)  # (B, N_anchors_all, 4 + 1 + N_class) N_anchors_all = 60*76 + 30*38 + 15*19=5985

            outputs = self.decode_outputs(outputs, hw, dtype=outputs[0][0].type())

            return outputs, ((reg1, reg2, reg3), (obj1, obj2, obj3), (cls1, cls2, cls3))

    def decode_outputs(self, outputs: Tensor, hw: List[Tuple[int, int]], dtype: torch.dtype) -> Tensor:
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(hw, self.strides):
            yv, xv = torch.meshgrid(torch.arange(hsize, device=outputs.device),
                                    torch.arange(wsize, device=outputs.device), indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            # shape = grid.shape[:2]
            # strides.append(torch.full((*shape, 1), stride))
            strides.append(torch.full((1, hsize * wsize, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)  # (0,0), (0,1), (0,2) ...
        strides = torch.cat(strides, dim=1).type(dtype)  # (32,32,32)... (16,16,16), (8,8,8)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides  # cx, cy
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides  # wh
        return outputs  # (cx, cy, w, h, obj, cls)


class Detect_YoloCS(Detect_Yolox):
    def __init__(self, num_classes, feat_channels=256, stacked_convs=0,in_channels=[128, 256, 512], use_stem=False):
        super().__init__(num_classes, feat_channels, stacked_convs,in_channels, use_stem)
        self.n_anchors = 1
        self.num_classes = num_classes
        # self.stacked_convs = 0

        self.levels = nn.ModuleList()
        for i in range(len(in_channels)):
            self.levels.append(
                BBoxHead_CS(
                    num_classes, self.n_anchors, in_channels[i], feat_channels,
                    act="silu", depthwise=False, use_stem=use_stem, stacked_convs=stacked_convs))

        print('Use Detect_YoloCS')


class Detect_YoloCS2(Detect_Yolox):
    def __init__(self, num_classes, feat_channels=256, stacked_convs=0,in_channels=[128, 256, 512], use_stem=False):
        super().__init__(num_classes, feat_channels, stacked_convs,in_channels, use_stem)
        self.n_anchors = 1
        self.num_classes = num_classes
        # self.stacked_convs = 0

        self.levels = nn.ModuleList()
        for i in range(len(in_channels)):
            self.levels.append(
                BBoxHead_CS2(
                    num_classes, self.n_anchors, in_channels[i], feat_channels,
                    act="silu", depthwise=False, use_stem=use_stem, stacked_convs=stacked_convs))
        print('Use Detect_YoloCS2')





class Detect_YoloCS3(Detect_Yolox):
    def __init__(self, num_classes, feat_channels=256, stacked_convs=0,in_channels=[128, 256, 512], use_stem=False):
        super().__init__(num_classes, feat_channels, stacked_convs,in_channels, use_stem)
        self.n_anchors = 1
        self.num_classes = num_classes
        # self.stacked_convs = 0

        self.levels = nn.ModuleList()
        for i in range(len(in_channels)):
            self.levels.append(
                BBoxHead_CS3(
                    num_classes, self.n_anchors, in_channels[i], feat_channels,
                    act="silu", depthwise=False, use_stem=use_stem, stacked_convs=stacked_convs))
        print('Use Detect_YoloCS3')




class DetectionModel(nn.Module):
    def __init__(self, cfg_Detection:Dict):
        super().__init__()
        ch = cfg_Detection['input_channel']
        self.model, self.save = parse_model(cfg_Detection, [ch])
        self.inplace = True
        m = self.model[-1]
        if isinstance(m, Detect_Yolo):
            # H, W = 60, 76  # 因为在SNN中，nuron的个数在第一次调用时就已经初始化好了，所以要保证这里的H,W和输入给DetectionModel中的大小一致
            # m.inplace = self.inplace
            # forward = lambda x: self.forward(x)
            # m.stride = torch.tensor([W / x.shape[-2] for x in forward(torch.zeros(1, ch, H, W))])  # forward, m.stride = [8., 16., 32.]
            # m.stride = m.stride * 4 # because we downsample the image in the attention module
            
            # 下面m.stride写死了，因为如果不写死的话，在网络初始化的时候，会调用一次dectedion
            # 但是这一次detection是在cpu上运行的，会导致backward的时候，出现not in same device的错误
            # 如果需要更改m.stride， 运行上面注释掉的代码即可

            if len(m.anchors) == 3:
                m.strides = torch.Tensor([4.0, 8.0, 16.0])
            if len(m.anchors) == 2:
                m.strides = torch.Tensor([8.0, 16.0])

            check_anchor_order(m)
            m.anchors /= m.strides.view(-1, 1, 1) # m.anchors.shape [3,3,2]
            self.strides = m.strides
            self._initialize_biases_yolo()  # only run once

        if isinstance(m, Detect_Yolox):
            m.strides = torch.Tensor([4.0, 8.0, 16.0])
            self.strides = m.strides
            self._initialize_biases_yolox(prior_prob=1e-2)

    def forward_backbone(self, x, profile=False):
        # input x , shape (bs, C, H , W)
        y, dt = [], []  # outputs
        y.append(x)  # layer 0 input
        for m in self.model[:-1]:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            # print(m.__class__.__name__ + f' x mean:{x.mean()}')
            y.append(x if m.i in self.save else None)  # save output
        # return for detection head
        m = self.model[-1]
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        return x


    def forward_detect(self, x, profile=False):
        m = self.model[-1]
        x = m(x)  # Detect()
        return x


    def forward(self, x, profile=False, visualize=False, last_step=False, step_mode='s'):
        if step_mode == 's':
            # input x , shape (bs, C, H , W)
            y, dt = [], []  # outputs
            y.append(x)  # layer 0 input
            for m in self.model[:-1]:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)  # run
                # print(m.__class__.__name__ + f' x mean:{x.mean()}')
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    pass
                    # feature_visualization(x, m.type, m.i, save_dir=visualize)

            # Detect head
            if last_step: # the last_step for single step mode
                m = self.model[-1]
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                x = m(x)  # Detect()
            return x
        elif step_mode == 'm':
            # input x , shape (T, bs, C, H , W)
            y, dt = [], []  # outputs
            y.append(x)  # layer 0 input
            for m in self.model[:-1]:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                x = m(x)  # run
                # print(x.shape)
                y.append(x if m.i in self.save else None)  # save output

            # Detect head
            m = self.model[-1]
            x = y[m.f][-1,...] if isinstance(m.f, int) else [x[-1,...] if j == -1 else y[j][-1,...] for j in m.f]
            x = m(x)  # Detect()
            return x


    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases_yolo(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.strides):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_yolox(self, prior_prob):
        m = self.model[-1]
        for level in m.levels:
            b = level.cls_pred.bias.view(1,-1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            level.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            b = level.obj_pred.bias.view(1,-1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            level.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def parse_model(d, ch):
    # d: model dict
    # ch: input channels
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    layers, save, c2 = [], [0], ch[-1]  # layers, savelist, ch out
    # layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]): # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                # args[j] = eval(a) if isinstance(a, str) else a  # eval strings\
                args[j] = a
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {LCB, CBL, Conv_BN,
                 Bottleneck_LCB, Bottleneck_CBL, Bottleneck_Res,
                 C3_LCB, C3_CBL, C3_Res,
                 C3_Mp_LCB, C3_Mp_CBL, C3_Mp_Res}:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
            if m in {C3_LCB, C3_CBL, C3_Res,
                 C3_Mp_LCB, C3_Mp_CBL, C3_Mp_Res}:
                args.insert(2,n) # number of repeats
                n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {UpSample, DownSample}:
            c2 = ch[f]
            # c1 = c2
            # args = [*args, c1, c2]
        elif m in {Mp_layer, LIF_layer}:
            c2 = ch[f]
            args = [c2, *args]
        elif m in {Detect_Yolo}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m in {Detect_Yolox,Detect_YoloCS,Detect_YoloCS2,Detect_YoloCS3}:
            args.append([ch[x] for x in f])


        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i+1, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i+1:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        # different from original yolo v5, we take the input as the 0 layer, and the output of first layer as 1
        save.extend(x % (i+1) for x in ([f] if isinstance(f, int) else f) if x != -1 and x != 0)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv3 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.strides[-1] - m.strides[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f"Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)