from typing import Any

import torch
import torch.nn as nn
import lightning as L
from torch import Tensor

from utils.common import parse_label_data_yolo, parse_event_data, postprocess
from spikingjelly.activation_based import functional

# from utils.psee_toolbox.io.box_filtering import filter_boxes_eval
# from utils.psee_toolbox.metrics.coco_utils import coco_eval



from utils.torch_utils import smart_optimizer
from utils.plots import plot_lr_scheduler
from utils.common import non_max_suppression

from GEN1_od.models_spiking.val import run_batch_metrics, compute_metrics
import time

class Gen1DetectionModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        cfg_embed = cfg['cfg_embed']
        cfg_attention = cfg['cfg_attention']
        cfg_latent_memory = cfg['cfg_latent_memory']
        cfg_Detection = cfg['cfg_Detection']
        cfg_pretrain = cfg['cfg_pretrain']
        self.cfg_loss = cfg['cfg_loss']
        self.cfg_optimizer = cfg['cfg_optimizer']
        self.cfg_lr = cfg['cft_lr']
        exec_string = cfg['exec_string']
        self.cfg_freeze_warm_up = cfg['cfg_freeze_warm_up']
        self.cfg_freeze_permanent = cfg['cfg_freeze_permanent']

        self.nc = cfg_Detection['nc']
        self.names = cfg['cfg_names']

        step_mode = cfg['cfg_Detection']['step_mode'] if 'step_mode' in cfg['cfg_Detection'] else 'SingleStep'

        if step_mode == 'SingleStep':
            from GEN1_od.models_spiking.model_single_step import Gen1Spiking
            self.model = Gen1Spiking(cfg_embed, cfg_attention, cfg_latent_memory, cfg_Detection, cfg_pretrain, exec_string)

        elif step_mode == 'MultiStep':
            from GEN1_od.models_spiking.model_multi_step import Gen1Spiking
            self.model = Gen1Spiking(cfg_embed, cfg_attention, cfg_latent_memory, cfg_Detection, cfg_pretrain, exec_string)
            # functional.set_step_mode(self.model.detection, 'm')
        else:
            raise ValueError(f'Unknown step_mode: {step_mode}')



        detection_head_name = self.model.detection.model[-1].__class__.__name__
        if detection_head_name == 'Detect_Yolo':
            from GEN1_od.models_spiking.loss_yolo import ComputeLoss_Yolo
            self.compute_loss = ComputeLoss_Yolo(self.model, self.cfg_loss)
            print('Using Yolo loss')
        elif detection_head_name in ['Detect_Yolox', 'Detect_YoloCS', 'Detect_YoloCS2', 'Detect_YoloCS3']:
            from GEN1_od.models_spiking.loss_yolox import ComputeLoss_Yolox
            self.compute_loss = ComputeLoss_Yolox(self.model, self.cfg_loss)
            print('Using Yolox loss')


        self.val_stats = None
        self.release_freeze_warm_up = False
        self.freeze_warm_up_init_finished = False
        self.freeze_permanent_init_finished = False





    def forward(self, events_list) -> Tensor:
        return self.model(events_list)


    def training_step(self, batch, batch_idx):


        # ==============warm up================
        if 'warm_up_steps' in self.cfg_optimizer:
            import numpy as np
            nb = len(self.trainer.train_dataloader) # number of batches per epoch
            nw = max(round(self.cfg_optimizer['warm_up_epoches'] * nb), self.cfg_optimizer['warm_up_steps'])
            # print(f'global step: {self.global_step}, warm up steps: {nw}')
            ni = self.global_step
            if ni <= nw:
                xi = [0,nw]
                for j, x in enumerate(self.optimizers().param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    if j == 0 or j == 1: # all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [0.0, x['initial_lr']])
                    elif j == 2:  # bias lr falls from 0.1 to lr0
                        x['lr'] = np.interp(ni, xi, [self.cfg_optimizer['warm_up_bias_lr'], x['initial_lr']])
                    elif j == 3: # tau lr falls from 0.1 to lr0
                        x['lr'] = np.interp(ni, xi, [self.cfg_optimizer['warm_up_tau_lr'], x['initial_lr']])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.cfg_optimizer['warm_up_momentum'], self.cfg_optimizer['momentum']])
                    # print('\n-----')
                    # print(x['lr'])
                # ------------------------------
                if not self.freeze_warm_up_init_finished:
                    if self.cfg_freeze_warm_up is not None and len(self.cfg_freeze_warm_up) > 0:
                        for k, v in self.model.named_parameters():
                            for ff in self.cfg_freeze_warm_up:
                                if all(x in k for x in ff):
                                    v.requires_grad = False
                                    print(f'Freeze warm up: {k}')
                    self.freeze_warm_up_init_finished = True

            else: # ni >= nw
                if not self.release_freeze_warm_up:
                    if self.cfg_freeze_warm_up is not None and len(self.cfg_freeze_warm_up) > 0:
                        for k, v in self.model.named_parameters():
                            for ff in self.cfg_freeze_warm_up:
                                if all(x in k for x in ff):
                                    v.requires_grad = True
                        print(f'Release freeze warm up')
                    self.release_freeze_warm_up = True

                if not self.freeze_permanent_init_finished:
                    for k, v in self.model.named_parameters():
                        for ff in self.cfg_freeze_permanent:
                            if all(x in k for x in ff):
                                v.requires_grad = False
                                print(f'Freeze permanent: {k}')

                    self.freeze_permanent_init_finished = True
        # =======================================



        events_list, label, _, _ = batch


        events_list = parse_event_data(events_list)
        targets, valid_mask = parse_label_data_yolo(label) #output label:[batch_index, class, n_cx, n_cy, n_w, n_h]
        output = self.model(events_list) # yolo: output [(bs,3,60,76,7), (bs,3,30,38,7), (bs,3,15,19,7)]
                                         # yolox: output: (bs, 4, 60 ,76), (bs, 4, 30, 38), (bs, 4, 15, 19)
        #                                #                (bs, 1, 60, 76), (bs, 1, 30, 38), (bs, 1, 15, 19)
        #                                #                (bs, nc, 60, 76), (bs, nc, 30, 38), (bs, nc, 15, 19)
        loss, loss_items = self.compute_loss(output, targets, valid_mask)
        functional.reset_net(self.model)





        bs = len(events_list[0])
        if batch_idx % 1 == 0:
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
            self.log('ls_iou', loss_items[0], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
            self.log('ls_obj', loss_items[1], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
            self.log('ls_cls', loss_items[2], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)

        return loss

    def validation_step(self, batch, batch_idx):
        events_list, label, _, event_path = batch
        events_list = parse_event_data(events_list)
        targets, valid_mask = parse_label_data_yolo(label)  # output label:[batch_index, class, n_cx, n_cy, n_w, n_h]
        preds, outputs = self.model(events_list) # pred (cx,cy, w,h, obj_conf, class...)
        loss, loss_items = self.compute_loss(outputs, targets, valid_mask)
        functional.reset_net(self.model)

        bs, img_h, img_w = len(events_list[0]), 240, 304
        # after postprocess, yolox: (xl, yl, xr, yr, obj_conf, class_conf, class)
        preds = postprocess(preds, self.nc, conf_thre=0.1, nms_thre=0.5, class_agnostic=False)
        targets = filter_targets(targets, valid_mask)
        preds = filter_preds(preds, min_box_diag=30, min_box_side=10)

        iouv = torch.linspace(0.5, 0.95, 10, device=targets.device)  # iou vector for mAP@0.5:0.95
        self.val_stats = run_batch_metrics(preds, targets, iouv, self.val_stats, img_h, img_w)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)


        plot = False
        if plot:
            import numpy as np
            from utils.psee_toolbox.visualize import vis_utils as vis
            import cv2
            # only plot one
            bi = 0 # the batch index
            # convert events to image
            events_temp = [x[bi] for x in events_list]
            events_temp = torch.cat(events_temp, dim=0)
            events_np = events_temp.cpu().numpy()
            dtype = [('t', '<u4'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')]
            events_np_structured = np.empty(events_np.shape[0], dtype=dtype)
            events_np_structured['t'] = events_np[:, 0]
            events_np_structured['x'] = events_np[:, 1]
            events_np_structured['y'] = events_np[:, 2]
            events_np_structured['p'] = events_np[:, 3]
            print('\n', event_path[0], '\n')
            print(f'Number of events: {events_np_structured.shape[0]} \n')
            print(label[0]['valid'])
            img = vis.make_binary_histo(events_np_structured, img=None)

            targets = targets[targets[:, 0] == bi]

            # plot the ground truth
            for indx in range(targets.shape[0]):  # plot the ground truth
                cx_n, cy_n, w_n, h_n = targets[indx][2:]
                (cx, cy, w, h) = (cx_n * img_w, cy_n * img_h, w_n * img_w, h_n * img_h)
                pt1 = (int(cx - w / 2), int(cy - h / 2))
                pt2 = (int(cx + w / 2), int(cy + h / 2))
                cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)  # green

            current_preds = preds[bi]
            if current_preds is not None:
                for indx in range(current_preds.shape[0]):
                    xl, yl, xr, yr = current_preds[indx][0], current_preds[indx][1], current_preds[indx][2], \
                    current_preds[indx][3]
                    pt1 = (int(xl), int(yl))
                    pt2 = (int(xr), int(yr))
                    obj_conf = current_preds[indx][4].item()
                    obj_conf_str = "{:.2f}".format(obj_conf)
                    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)  # red
                    cv2.putText(img, obj_conf_str, (int(pt2[0] - 10), int(pt2[1] + 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('out', img)
            cv2.waitKey(0)
            pass




    def on_train_start(self) -> None:
        self.compute_loss.__init__(self.model, self.cfg_loss)
        self.release_freeze_warm_up = False
        self.freeze_warm_up_init_finished = False
        self.freeze_permanent_init_finished = False

    # def on_train_epoch_start(self) -> None:
    #     self.epoch_start_time = time.time()
    #     print('===============train epoch=========================')
    #
    # def on_train_epoch_end(self) -> None:
    #     epoch_end_time = time.time()
    #     print(f'\nEpoch time: {epoch_end_time - self.epoch_start_time}')

    def on_validation_start(self) -> None:
        self.compute_loss.__init__(self.model, self.cfg_loss)



    def on_validation_epoch_start(self) -> None:
        self.val_stats = []

    def on_validation_epoch_end(self) -> None:
        map50, map = compute_metrics(self.val_stats, names=self.names)
        self.log('map50', map50, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('map', map, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)



    def configure_optimizers(self):
        optimizer = smart_optimizer(self.model, self.cfg_optimizer['optimizer'], self.cfg_optimizer['lr0'], self.cfg_optimizer['momentum'], self.cfg_optimizer['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg_lr['epochs'])

        return [optimizer], [scheduler]

def filter_targets(targets, valid_mask):
    targets = targets[(valid_mask[:,1]).bool(),:]
    # if targets.shape[0] == 0:
    #    print('a')
    return targets

def filter_preds(preds, min_box_diag=30, min_box_side=10):
    # remove small preds xl, yl, xr, yr
    for i in range(len(preds)):
        if preds[i] is None:
            continue
        W = preds[i][:, 2] - preds[i][:, 0]
        H = preds[i][:, 3] - preds[i][:, 1]
        diag_squre = W ** 2 + H ** 2
        min_side = torch.minimum(W, H)
        valid_mask = torch.logical_and(diag_squre >= min_box_diag ** 2, min_side >= min_box_side)
        preds[i] = preds[i][valid_mask, :]
    return preds
