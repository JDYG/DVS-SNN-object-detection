import glob
import os
import logging
import logging.config
import torch
import numpy as np
import time
import torchvision
from utils.metrics import box_iou, fitness

# Get file list
def get_list(dpath, ext, root=None):
    if isinstance(dpath, (list, tuple)):
        fpathlist = dpath
    elif os.path.isdir(dpath):
        if isinstance(ext, str):
            ext = [ext]
        fpathlist = []
        for e in ext:
            fpathlist += glob.glob(f'{dpath}/*.{e}')
        fpathlist = sorted(fpathlist)
    elif os.path.isfile(dpath):
        fpathlist = [fpath for fpath in open(dpath, 'r').read().split('\n') if fpath != '']
    else:
        fpathlist = []

    if root is not None:
        strip = lambda x: x[1:] if x[0] == '.' else x
        fpathlist = [ root + '/' + strip(fpath) for fpath in fpathlist ]

    return fpathlist

import os

def mkdir(dpath):
    # if exists, do nothing
    if os.path.exists(dpath):
        # print('The directory already exists: {}'.format(dpath))
        pass
    else:
        os.makedirs(dpath, exist_ok=False)
        print('Successfully create directory: {}'.format(dpath))



def parse_label_data(label):
    out_gt_bboxes = []
    out_gt_labels = []
    out_ignore_masks = []
    for _, l in enumerate(label):
        out_gt_bboxes.append(l['bboxes'])
        out_gt_labels.append(l['labels'])
        out_ignore_masks.append(l['ignore_mask'])
    return out_gt_bboxes, out_gt_labels, out_ignore_masks


# parse the label for yolo detection
# the input label是size 为batch size大小的list, 每个list里面是一个dict{'times':, 'bboxes':, 'labels':, 'ignore_mask':}
# the input label is xl, yl, xr, yr
# the output targets:[batch_index, class, n_cx, n_cy, n_w, n_h]
def parse_label_data_yolo(label, image_w = 304, image_h = 240):
    out_targets = []
    valid_mask = []
    for bs_index, l in enumerate(label):
        n_bboxes = len(l['labels'])
        device = l['bboxes'].device
        bi = torch.ones((n_bboxes,1), device = device) * bs_index
        labels_temp = torch.cat(( (l['labels']).unsqueeze(-1) , l['bboxes']), dim=-1)
        out_targets.append(torch.cat((bi, labels_temp), dim=-1))
        valid_mask.append(torch.cat((bi, (l['valid']).unsqueeze(-1)), dim=-1))

    out_targets = torch.cat(out_targets, dim=0) # [batch_index, class, xl, yl, xr, yr]
    valid_mask = torch.cat(valid_mask, dim=0) # [batch_index, valid]
    temp_box = xyxy2xywhn(out_targets[:,2:], image_w, image_h, clip=False) # has been clipped in the dataloader
    out_targets = torch.cat((out_targets[:,0:2], temp_box), dim=-1) # [batch_index, class, n_cx, n_cy, n_w, n_h]
    return out_targets, valid_mask

def xyxy2xywhn(x, w, h, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # convert y to float
    y = y.float()
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def cxcywh_n2xlylxryr(x, img_h, img_w):
    # Convert nx4 boxes from [n_cx, n_cy, n_w, n_h] to [xl, yl, xr, yr]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] - x[..., 2] / 2) * img_w  # top left x
    y[..., 1] = (x[..., 1] - x[..., 3] / 2) * img_h  # top left y
    y[..., 2] = (x[..., 0] + x[..., 2] / 2) * img_w  # bottom right x
    y[..., 3] = (x[..., 1] + x[..., 3] / 2) * img_h  # bottom right y
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


# parse event data 的作用是将原来存储在字典中的数据('events:' tensor(xxxx))转换成tensor
def parse_event_data(inputs):
    def _nested_shape(lst, shape=[]):
        if isinstance(lst, (list, tuple)):
            shape += [len(lst)]
            return _nested_shape(lst[0], shape)
        else:
            return shape

    datas = inputs
    shape = _nested_shape(datas)

    if len(shape) == 2:
        list_events = []
        for _, ds in enumerate(datas):
            events = parse_event_data(ds)
            list_events.append(events)

        return list_events
    else:
        if isinstance(datas[0], dict):
            # in case of raw events --> batch of event tensors
            events = [ d['events'] for d in datas ]
        else:
            # in case of event frames --> tensor with shape (B, C, H, W)
            events = datas
        return events


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # input prediction: (B, N_anchors_all, 4 + 1 + N_class) N_anchors_all = 60*76 + 30*38 + 15*19=5985
    # input prediction (cx, cy, w, h )
    # output (xl, yl, xr, yr, obj_conf, class_conf, class_pred)
    import torchvision
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4] # cx, cy, w, h -> xl, yl, xr, yr

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # input prediction的形状为[1, 18900, 85], nc =80
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv3 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates # 最后一个维度第4个值代表该区域是否有物体的置信度，因此这里是筛选出可能有物体的区域

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference, 对每个batch中的图片进行迭代
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence， 选择可能有物体的区域（大于obj置信度）

        
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):# 不进入此逻辑
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]: # 如果没有物体，直接跳过
            continue

        # Compute conf
        # 将置信度乘以代表物体类别的score，并赋给物体类别score
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ?? NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output



# https://github.com/ultralytics/yolov5/blob/master/utils/general.py
LOGGING_NAME = "YOLOv5"  # logging name
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {
                name: {
                    "level": level,
                    "handlers": [name],
                    "propagate": False,
                }
            },
        }
    )
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]
