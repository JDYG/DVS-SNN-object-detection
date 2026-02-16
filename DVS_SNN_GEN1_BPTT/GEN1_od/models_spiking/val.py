import torch
import numpy as np

from utils.metrics import ConfusionMatrix, box_iou, ap_per_class
from utils.common import cxcywh_n2xlylxryr
# from utils.common import non_max_suppression


# 这个函数主要有两个作用：

# 作用1：对预测框与gt进行匹配
# 作用2：对匹配上的预测框进行iou数值判断，用True来填充，其余没有匹配上的预测框的所以行数全部设置为False
# 对于每张图像的预测框，需要筛选出能与gt匹配的框来进行相关的iou计算，设置了iou从0.5-0.95的10个梯度，
# 如果匹配的预测框iou大于相对于的阈值，则在对应位置设置为True，否则设置为False；而对于没有匹配上的预测框全部设置为False。

# Q：为什么要筛选？
# 这是因为一个gt只可能是一个类别，不可能是多个类别，所以需要取置信度最高的类别进行匹配。
# 但是此时还可能多个gt和一个预测框匹配，同样的，为这个预测框分配iou值最高的gt，依次来实现一一配对

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix 返回每个预测框在10个IoU阈值上是TP还是FP
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    # 构建一个[pred_nums, 10]全为False的矩阵
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    # 计算每个gt与每个pred的iou，shape为: [gt_nums, pred_nums]
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match # iou超过阈值而且类别正确，则为True，返回索引
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                # argsort获得由小到大排序的索引, [::-1]相当于取反reserve操作，变成由大到小排序的索引，对matches矩阵进行排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                '''
                参数return_index=True：表示会返回唯一值的索引，[0]返回的是唯一值，[1]返回的是索引
                matches[:, 1]：这里的是获取iou矩阵每个预测框的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
                这个操作的含义：每个预测框最多只能出现一次，如果有一个预测框同时和多个gt匹配，只取其最大iou的一个
                '''
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                '''
                matches[:, 0]：这里的是获取iou矩阵gt的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
                这个操作的含义: 每个gt也最多只能出现一次，如果一个gt同时匹配多个预测框，只取其匹配最大的那一个预测框
                '''
            correct[matches[:, 1].astype(int), i] = True
            # 在correct中，只有与gt匹配的预测框才有对应的iou评价指标，其他大多数没有匹配的预测框都是全部为False
            '''
            当前获得了gt与预测框的一一对应，其对于的iou可以作为评价指标，构建一个评价矩阵
            需要注意，这里的matches[:, 1]表示的是为对应的预测框来赋予其iou所能达到的程度，也就是iouv的评价指标
            '''
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def run_batch_metrics(preds, targets, iouv, stats, img_h, img_w):
    # input preds: preds after NMS, the preds type: (n,6) tensor per image [xlylxryr, conf, cls]
    # input targets: targets type: (n,5) tensor per image [batch_index, cls, cx_n, cy_n, w_n, h_n]
    niou = iouv.numel()
    device = targets.device
    for si, pred in enumerate(preds):
        labels = targets[targets[:, 0] == si, 1:]  # [cls, cx_n, cy_n, w_n, h_n]
        nl = labels.shape[0]  # number of labels
        npr = pred.shape[0]  if pred is not None else 0
        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
        if npr == 0:  # if no prediction
            if nl:  # when no prediction, but has targets
                stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
            continue
        if pred.shape[1] == 7:  # if has obj_conf, class_conf, convert conf = obj_conf * class_conf
            pred[:, 4] *= pred[:, 5]
            pred[:, 5] = pred[:, 6]
            pred = pred[:, :6]  # remove class_conf

        if nl:  # when having predictions and having targets
            # predn [xl,yl,xr,yr, conf, cls]
            # labelsn [cls, xl, yl, xr, yr]
            predn = pred.clone()
            tbox = cxcywh_n2xlylxryr(labels[:,1:], img_h=img_h, img_w=img_w)
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)

        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
    return stats

def compute_metrics(stats, names):
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir='.', names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    # nt = np.bincount(stats[3].astype(int), minlength=2)  # number of targets per class, minlength is the number of classes

    return map50, map