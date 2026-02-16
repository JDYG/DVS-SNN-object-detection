import torch
import torch.nn as nn
from GEN1_od.models_spiking.neurons import MpLIFNode, Mp_ParametricLIFNode
from spikingjelly.activation_based import neuron, layer
from visualizer import get_local

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def create_neuron(tau, v_threshold, neuron_type, **kwargs):
    if neuron_type == 'LIF':
        return neuron.LIFNode(tau=tau, v_threshold=v_threshold,**kwargs)
    elif neuron_type == 'PLIF':
        return neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, **kwargs)
    elif neuron_type == 'Mp_LIF':
        return MpLIFNode(tau=tau, v_threshold=v_threshold, **kwargs)
    elif neuron_type == 'Mp_PLIF':
        return Mp_ParametricLIFNode(init_tau=tau, v_threshold=1.0, decay_input=False, **kwargs)
    elif neuron_type == 'None':
        return nn.Identity()
    else:
        raise NotImplementedError('Neuron type not implemented.')

def create_batchnrom_2d(num_features, batchnorm_type, T=None, step_mode='s'):
    if batchnorm_type == 'BN':
        return layer.BatchNorm2d(num_features, step_mode=step_mode)
    elif batchnorm_type == 'tdBN':
        if step_mode == 's':
            raise ValueError('tdBN can only be used in multi step mode.')
        return layer.ThresholdDependentBatchNorm2d(alpha=1.0, v_th=1.0, num_features=num_features)
    elif batchnorm_type == 'TEBN':
        if T is None:
            raise ValueError('T must be specified when using TemporalEffectiveBatchNorm2d.')
        return layer.TemporalEffectiveBatchNorm2d(T=T, num_features=num_features, step_mode=step_mode)
    elif batchnorm_type == 'None':
        return nn.Identity()


class LCB(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, neuron_type='LIF', tau = 1.5, bn_type = 'BN', step_mode = 's', T=None):
        super().__init__()
        self.conv = layer.Conv2d(c1, c2, k, s, autopad(k,p,d), bias=False, step_mode=step_mode)
        self.bn = create_batchnrom_2d(c2, bn_type, T, step_mode)
        self.sn = create_neuron(tau, 1.0, neuron_type, step_mode=step_mode)

     # @get_local('LCB_fr')
    # get_local('')
    def forward(self, x):

        return self.bn(self.conv(self.sn(x)))

class CBL(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, neuron_type='LIF', tau = 1.5, bn_type = 'BN', step_mode = 's', T=None):
        super().__init__()
        self.conv = layer.Conv2d(c1, c2, k, s, autopad(k,p,d), bias=False, step_mode=step_mode)
        self.bn = create_batchnrom_2d(c2, bn_type, T, step_mode)
        self.sn = create_neuron(tau, 1.0, neuron_type, step_mode=step_mode)

    def forward(self, x):
        return self.sn(self.bn(self.conv(x)))



class Bottleneck_LCB(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, e = 0.5, neuron_type='LIF', tau = 1.5, bn_type = 'BN', step_mode = 's', T=None):
        super().__init__()
        c_ = int(out_channels * e)

        self.cv1 = LCB(in_channels, c_, 1, 1, None,1, neuron_type, tau, bn_type, step_mode, T)
        self.cv2 = LCB(c_, out_channels, 3, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_CBL(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, e = 0.5, neuron_type='LIF', tau = 1.5, bn_type = 'BN', step_mode = 's', T=None):
        super().__init__()
        c_ = int(out_channels * e)

        self.cv1 = CBL(in_channels, c_, 1, 1, None,1, neuron_type, tau, bn_type, step_mode, T)
        self.cv2 = CBL(c_, out_channels, 3, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_Res(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, e = 0.5, neuron_type='LIF', tau = 1.5, bn_type = 'BN', step_mode = 's', T=None):
        super().__init__()
        c_ = int(out_channels * e)

        self.cv1 = CBL(in_channels, c_, 1, 1, None,1, neuron_type, tau, bn_type, step_mode, T)
        self.cv2 = CBL(c_, out_channels, 3, 1, None, 1, 'None', tau, bn_type, step_mode, T)
        self.add = shortcut and in_channels == out_channels
        self.sn = create_neuron(tau, 1.0, neuron_type, step_mode=step_mode)

    def forward(self, x):
        if self.add:
            return self.sn(x+self.cv2(self.cv1(x)))
        else:
            return self.sn(self.cv2(self.cv1(x)))


class C3_LCB(nn.Module):
    def __init__(self, in_channels, out_channels, n = 1, shortcut=True, e=0.5, neuron_type='LIF', tau=1.5, bn_type='BN', step_mode='s', T=None):
        super().__init__()
        c_ = int(out_channels * e)
        self.cv1 = LCB(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv2 = LCB(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv3 = LCB(2 * c_, out_channels, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.m = nn.Sequential(*(Bottleneck_LCB(c_, c_, shortcut, 1.0, neuron_type, tau, bn_type, step_mode, T) for _ in range(n)))
        self.step_mode = step_mode
    def forward(self, x):
        y1 = self.cv2(x)
        y2 = self.m(self.cv1(x))
        if self.step_mode == 's':
            return self.cv3(torch.cat((y1, y2), dim=1))
        elif self.step_mode == 'm':
            return self.cv3(torch.cat((y1, y2), dim=2))



class C3_CBL(nn.Module):
    def __init__(self, in_channels, out_channels, n = 1, shortcut=True, e=0.5, neuron_type='LIF', tau=1.5, bn_type='BN', step_mode='s', T=None):
        super().__init__()
        c_ = int(out_channels * e)
        self.cv1 = CBL(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv2 = CBL(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv3 = CBL(2 * c_, out_channels, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.m = nn.Sequential(*(Bottleneck_CBL(c_, c_, shortcut, 1.0, neuron_type, tau, bn_type, step_mode, T) for _ in range(n)))
        self.step_mode = step_mode
    def forward(self, x):
        y1 = self.cv2(x)
        y2 = self.m(self.cv1(x))
        if self.step_mode == 's':
            return self.cv3(torch.cat((y1, y2), dim=1))
        elif self.step_mode == 'm':
            return self.cv3(torch.cat((y1, y2), dim=2))


class C3_Res(nn.Module):
    def __init__(self, in_channels, out_channels, n = 1, shortcut=True, e=0.5, neuron_type='LIF', tau=1.5, bn_type='BN', step_mode='s', T=None):
        super().__init__()
        c_ = int(out_channels * e)
        self.cv1 = CBL(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv2 = CBL(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv3 = CBL(2 * c_, out_channels, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.m = nn.Sequential(*(Bottleneck_Res(c_, c_, shortcut, 1.0, neuron_type, tau, bn_type, step_mode, T) for _ in range(n)))
        self.step_mode = step_mode
    def forward(self, x):
        y1 = self.cv2(x)
        y2 = self.m(self.cv1(x))
        if self.step_mode == 's':
            return self.cv3(torch.cat((y1, y2), dim=1))
        elif self.step_mode == 'm':
            return self.cv3(torch.cat((y1, y2), dim=2))


class C3_Mp_LCB(nn.Module):
    def __init__(self, in_channels, out_channels, n = 1, shortcut=True, e=0.5, neuron_type='LIF', tau=1.5, bn_type='BN', step_mode='s', T=None):
        super().__init__()
        c_ = int(out_channels * e)
        self.cv1 = LCB(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv2 = LCB(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv3 = LCB(2 * c_, out_channels, 1, 1, None, 1, 'Mp_LIF', tau, bn_type, step_mode, T)

        self.m = nn.Sequential(*(Bottleneck_LCB(c_, c_, shortcut, 1.0, neuron_type, tau, bn_type, step_mode, T) for _ in range(n)))

        self.step_mode = step_mode
    def forward(self, x):
        y1 = self.cv2(x)
        y2 = self.m(self.cv1(x))
        if self.step_mode == 's':
            return self.cv3(torch.cat((y1, y2), dim=1))
        elif self.step_mode == 'm':
            return self.cv3(torch.cat((y1, y2), dim=2))



class C3_Mp_CBL(nn.Module):
    def __init__(self, in_channels, out_channels, n = 1, shortcut=True, e=0.5, neuron_type='LIF', tau=1.5, bn_type='BN', step_mode='s', T=None):
        super().__init__()
        c_ = int(out_channels * e)
        self.cv1 = CBL(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv2 = CBL(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv3 = CBL(2 * c_, out_channels, 1, 1, None, 1, 'Mp_LIF', tau, bn_type, step_mode, T)

        self.m = nn.Sequential(*(Bottleneck_CBL(c_, c_, shortcut, 1.0, neuron_type, tau, bn_type, step_mode, T) for _ in range(n)))

        self.step_mode = step_mode
    def forward(self, x):
        y1 = self.cv2(x)
        y2 = self.m(self.cv1(x))
        if self.step_mode == 's':
            return self.cv3(torch.cat((y1, y2), dim=1))
        elif self.step_mode == 'm':
            return self.cv3(torch.cat((y1, y2), dim=2))



class C3_Mp_Res(nn.Module):
    def __init__(self, in_channels, out_channels, n = 1, shortcut=True, e=0.5, neuron_type='LIF', tau=1.5, bn_type='BN', step_mode='s', T=None):
        super().__init__()
        c_ = int(out_channels * e)
        self.cv1 = CBL(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv2 = CBL(in_channels, c_, 1, 1, None, 1, neuron_type, tau, bn_type, step_mode, T)
        self.cv3 = CBL(2 * c_, out_channels, 1, 1, None, 1, 'Mp_LIF', tau, bn_type, step_mode, T)

        self.m = nn.Sequential(*(Bottleneck_Res(c_, c_, shortcut, 1.0, neuron_type, tau, bn_type, step_mode, T) for _ in range(n)))

        self.step_mode = step_mode
    def forward(self, x):
        y1 = self.cv2(x)
        y2 = self.m(self.cv1(x))
        if self.step_mode == 's':
            return self.cv3(torch.cat((y1, y2), dim=1))
        elif self.step_mode == 'm':
            return self.cv3(torch.cat((y1, y2), dim=2))


class Mp_layer(nn.Module):
    def __init__(self, num_features, tau, neuron_type='Mp_LIF', bn_type='BN', T=None):
        super().__init__()
        self.sn = create_neuron(tau, 1.0, neuron_type, step_mode='s')
        self.bn = create_batchnrom_2d(num_features, bn_type, T, step_mode='s')

    def forward(self, x):
        return self.bn(self.sn(x))

class LIF_layer(nn.Module):
    def __init__(self, num_features, tau, neuron_type='LIF', bn_type='BN', T=None):
        super().__init__()
        self.sn = create_neuron(tau, 1.0, neuron_type, step_mode='s')
        self.bn = create_batchnrom_2d(num_features, bn_type, T, step_mode='s')

    def forward(self, x):
        return self.bn(self.sn(x))

class Conv_BN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, d =1, bn_type='BN', T=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k,p,d), bias=False)
        self.bn = create_batchnrom_2d(c2, bn_type, T, step_mode='s')
    def forward(self, x):
        return self.bn(self.conv(x))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class identity(nn.Identity):
    def __init__(self):
        super().__init__()

class DownSample(nn.Module):
    def __init__(self, method='avg',k=2,c1=None,c2=None):
        super().__init__()
        if method == 'avg':
            raise NotImplementedError('Downsample method not implemented.')
        elif method == 'max':
            self.downsample = layer.MaxPool2d(k)
        else:
            raise NotImplementedError('Downsample method not implemented.')
    def forward(self, x):
        return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, method='nearest',k=2, step_mode='s',c1=None,c2=None):
        super().__init__()
        if method == 'nearest':
            self.upsample = layer.Upsample(size=None, scale_factor=k, mode=method, step_mode=step_mode)


    def forward(self, x):
        return self.upsample(x)





# ============Yolox blocks================
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class BBoxHead(nn.Module):
    def __init__(self, num_classes, n_anchors, in_channels, feat_channels, act="silu", depthwise=False, use_stem=False,
                 stacked_convs=2):
        super().__init__()
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.n_anchors = n_anchors

        Conv = DWConv if depthwise else BaseConv

        if use_stem:
            self.stem = BaseConv(
                in_channels=int(in_channels),
                out_channels=feat_channels,
                ksize=1,
                stride=1,
                act=act)
        else:
            self.stem = nn.Identity()

        self.cls_conv = nn.Sequential(
            *[Conv(in_channels=feat_channels, out_channels=feat_channels, ksize=3, stride=1, act=act)
              for _ in range(stacked_convs)])

        self.reg_conv = nn.Sequential(
            *[Conv(in_channels=feat_channels, out_channels=feat_channels, ksize=3, stride=1, act=act)
              for _ in range(stacked_convs)])

        self.cls_pred = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=n_anchors * num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.reg_pred = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=n_anchors * 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.obj_pred = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=n_anchors * 1,
            kernel_size=1,
            stride=1,
            padding=0,
        )


    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        obj_pred = self.obj_pred(reg_feat)
        return reg_pred, obj_pred, cls_pred


class BBoxHead_CS(nn.Module):
    def __init__(self, num_classes, n_anchors, in_channels, feat_channels, act="silu", depthwise=False, use_stem=False,
                 stacked_convs=2):
        super().__init__()
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.n_anchors = n_anchors

        Conv = DWConv if depthwise else BaseConv

        use_stem = False
        if use_stem:
            self.stem = BaseConv(
                in_channels=int(in_channels),
                out_channels=feat_channels,
                ksize=1,
                stride=1,
                act=act)
        else:
            self.stem = nn.Identity()


        self.cls_conv = BaseConv(in_channels=int(in_channels), out_channels=int(in_channels), ksize=1, stride=1, act=act)
        self.obj_conv = nn.Sequential(
            Conv(in_channels=int(in_channels), out_channels=int(in_channels/8), ksize=3, stride=1, act=act),
            Conv(in_channels=int(in_channels/8), out_channels=int(in_channels/16), ksize=3, stride=1, act=act),
            Conv(in_channels=int(in_channels/16), out_channels=int(in_channels/32), ksize=3, stride=1, act=act),
        )

        self.reg_conv = BaseConv(in_channels=int(in_channels), out_channels=int(in_channels), ksize=1, stride=1, act=act)

        self.cls_pred = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=n_anchors * num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.reg_pred = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=n_anchors * 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.obj_pred = nn.Conv2d(
            in_channels=int(in_channels/32),
            out_channels=n_anchors * 1,
            kernel_size=1,
            stride=1,
            padding=0,
        )


    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        obj_feat = self.obj_conv(x)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        obj_pred = self.obj_pred(obj_feat)
        return reg_pred, obj_pred, cls_pred


class BBoxHead_CS2(nn.Module):
    def __init__(self, num_classes, n_anchors, in_channels, feat_channels, act="silu", depthwise=False, use_stem=False,
                 stacked_convs=2):
        super().__init__()
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.n_anchors = n_anchors

        Conv = DWConv if depthwise else BaseConv

        use_stem = False
        if use_stem:
            self.stem = BaseConv(
                in_channels=int(in_channels),
                out_channels=feat_channels,
                ksize=1,
                stride=1,
                act=act)
        else:
            self.stem = nn.Identity()


        self.cls_conv = BaseConv(in_channels=int(in_channels), out_channels=int(in_channels), ksize=1, stride=1, act=act)

        self.obj_conv0 = Conv(in_channels=int(in_channels), out_channels=int(in_channels/8), ksize=3, stride=1, act=act)
        self.obj_conv = nn.Sequential(
            nn.Identity(),
            Conv(in_channels=int(in_channels/8), out_channels=int(in_channels/16), ksize=3, stride=1, act=act),
            Conv(in_channels=int(in_channels/16), out_channels=int(in_channels/32), ksize=3, stride=1, act=act),
        )
        self.reg_conv = Conv(in_channels=int(in_channels/8), out_channels=int(in_channels/8), ksize=3, stride=1, act=act)


        self.cls_pred = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=n_anchors * num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.reg_pred = nn.Conv2d(
            in_channels=int(in_channels/8),
            out_channels=n_anchors * 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.obj_pred = nn.Conv2d(
            in_channels=int(in_channels/32),
            out_channels=n_anchors * 1,
            kernel_size=1,
            stride=1,
            padding=0,
        )


    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_conv(x)
        obj_feat0 = self.obj_conv0(x)
        obj_feat = self.obj_conv(obj_feat0)
        reg_feat = self.reg_conv(obj_feat0)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        obj_pred = self.obj_pred(obj_feat)
        return reg_pred, obj_pred, cls_pred


class BBoxHead_CS3(nn.Module):
    def __init__(self, num_classes, n_anchors, in_channels, feat_channels, act="silu", depthwise=False, use_stem=False,
                 stacked_convs=2):
        super().__init__()
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.n_anchors = n_anchors

        Conv = DWConv if depthwise else BaseConv

        use_stem = False
        if use_stem:
            self.stem = BaseConv(
                in_channels=int(in_channels),
                out_channels=feat_channels,
                ksize=1,
                stride=1,
                act=act)
        else:
            self.stem = nn.Identity()


        self.cls_conv = nn.Sequential(
            BaseConv(in_channels=int(in_channels), out_channels=int(in_channels/4), ksize=1, stride=1, act=act),
            Conv(in_channels=int(in_channels/4), out_channels=int(in_channels / 4), ksize=3, stride=1, act=act),
        )

        self.obj_conv0 = Conv(in_channels=int(in_channels), out_channels=int(in_channels/8), ksize=3, stride=1, act=act)
        self.obj_conv = nn.Sequential(
            nn.Identity(),
            Conv(in_channels=int(in_channels/8), out_channels=int(in_channels/16), ksize=3, stride=1, act=act),
            Conv(in_channels=int(in_channels/16), out_channels=int(in_channels/32), ksize=3, stride=1, act=act),
        )
        self.reg_conv = Conv(in_channels=int(in_channels/8), out_channels=int(in_channels/8), ksize=3, stride=1, act=act)


        self.cls_pred = nn.Conv2d(
            in_channels=int(in_channels / 4),
            out_channels=n_anchors * num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.reg_pred = nn.Conv2d(
            in_channels=int(in_channels/8),
            out_channels=n_anchors * 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.obj_pred = nn.Conv2d(
            in_channels=int(in_channels/32),
            out_channels=n_anchors * 1,
            kernel_size=1,
            stride=1,
            padding=0,
        )


    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_conv(x)
        obj_feat0 = self.obj_conv0(x)
        obj_feat = self.obj_conv(obj_feat0)
        reg_feat = self.reg_conv(obj_feat0)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        obj_pred = self.obj_pred(obj_feat)
        return reg_pred, obj_pred, cls_pred
