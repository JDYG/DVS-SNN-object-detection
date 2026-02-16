import os
import numpy as np
import torch
from utils.metrics import ConfusionMatrix, box_iou, ap_per_class
from utils.common import cxcywh_n2xlylxryr

def main():
    folder = 'GEN1_od/video_infer/config_yolox_20e3_5_64attn'
    files = os.listdir(folder)
    preds_files = [file for file in files if file.endswith('preds.txt')]

    stats = []
    iouv = torch.linspace(0.5, 0.95, 10)
    count = 0
    for preds_file in preds_files:
        # count += 1
        # if count > 130:
        #     break
        print(f'Processing {preds_file}')
        gt_file = preds_file.replace('preds', 'gt')
        preds = np.loadtxt(os.path.join(folder, preds_file)) # (time_index, xl, yl, xr, yr, obj_conf, class_conf, class)
        # if preds_file == '319_preds.txt':
        #     continue
        if preds.ndim == 1:
            preds = preds[np.newaxis, :]
        conf = preds[:, 5] * preds[:, 6]
        # remove preds with conf < 0.4
        preds = preds[conf >= 0.3]
        gt = np.loadtxt(os.path.join(folder, gt_file)) # (time_index, class, n_cx, n_cy, n_w, n_h)
        time_index_preds = preds[:, 0]
        unique_time_index_preds = np.unique(time_index_preds)
        preds_list = []
        for time_index in unique_time_index_preds:
            data_for_time_index = preds[time_index_preds == time_index]
            preds_list.append(data_for_time_index)

        val_stats = run_metrics(preds_list, gt, iouv, stats, 240, 304)

    map50, map = compute_metrics(val_stats, names={0: 'car', 1: 'pedestrian'})
    print(f'mAP@0.5: {map50}, mAP: {map}')




def process_batch(detections, labels, iouv):

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)

    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:

                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def run_metrics(preds_list, targets, iouv, stats, img_h, img_w):
    niou = iouv.numel()

    targets = torch.from_numpy(targets)
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(0)
    device = targets.device
    for preds in preds_list:
        time_index = int(preds[0, 0])
        pred = preds[:, 1:]
        pred = torch.from_numpy(pred)
        labels = targets[targets[:, 0] == time_index][:, 1:] # [cls, cx_n, cy_n, w_n, h_n]
        nl = labels.shape[0]  # number of labels
        npr = pred.shape[0] if pred is not None else 0
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
            tbox = cxcywh_n2xlylxryr(labels[:, 1:], img_h=img_h, img_w=img_w)
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            # import cv2
            # img = 255*np.ones((img_h, img_w, 3), dtype=np.uint8)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

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


if __name__ == '__main__':
    main()


