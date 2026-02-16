import torch
import numpy as np

from utils.metrics import ConfusionMatrix, box_iou, ap_per_class
from utils.common import cxcywh_n2xlylxryr





def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix 返回每个预测框在10个IoU阈值上是TP还是FP
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                '''
                参数return_index=True：表示会返回唯一值的索引，[0]返回的是唯一值，[1]返回的是索引
                matches[:, 1]：这里的是获取iou矩阵每个预测框的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
                这个操作的含义：每个预测框最多只能出现一次，如果有一个预测框同时和多个gt匹配，只取其最大iou的一个
                '''
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                '''
                matches[:, 0]：这里的是获取iou矩阵gt的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
                这个操作的含义: 每个gt也最多只能出现一次，如果一个gt同时匹配多个预测框，只取其匹配最大的那一个预测框
                '''
            correct[matches[:, 1].astype(int), i] = True
            '''
            当前获得了gt与预测框的一一对应，其对于的iou可以作为评价指标，构建一个评价矩阵
            需要注意，这里的matches[:, 1]表示的是为对应的预测框来赋予其iou所能达到的程度，也就是iouv的评价指标
            '''
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def run_batch_metrics(preds, targets, iouv, stats, img_h, img_w):
    niou = iouv.numel()
    device = targets.device
    for si, pred in enumerate(preds):
        labels = targets[targets[:, 0] == si, 1:]
        nl = labels.shape[0]
        npr = pred.shape[0]  if pred is not None else 0
        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
            continue
        if pred.shape[1] == 7:
            pred[:, 4] *= pred[:, 5]
            pred[:, 5] = pred[:, 6]
            pred = pred[:, :6]

        if nl:
            predn = pred.clone()
            tbox = cxcywh_n2xlylxryr(labels[:,1:], img_h=img_h, img_w=img_w)
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)
            correct = process_batch(predn, labelsn, iouv)

        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
    return stats

def compute_metrics(stats, names):
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir='.', names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    return map50, map