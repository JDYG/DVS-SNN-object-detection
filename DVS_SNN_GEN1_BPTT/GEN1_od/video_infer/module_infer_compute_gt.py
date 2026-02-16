'''
only compute the preds at gt time index
if you want to compute the preds at every time index, please use the module_infer_plot_video.py
'''

import torch
import torch.nn as nn
import lightning as L
from torch import Tensor

from utils.common import parse_event_data, postprocess
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
        self.config_name = cfg['config_name']

        step_mode = cfg['cfg_Detection']['step_mode'] if 'step_mode' in cfg['cfg_Detection'] else 'SingleStep'

        if step_mode == 'SingleStep':
            from GEN1_od.video_infer.model_single_step_infer import Gen1Spiking
            self.model = Gen1Spiking(cfg_embed, cfg_attention, cfg_latent_memory, cfg_Detection, cfg_pretrain, exec_string)





        # detection_head_name = self.model.detection.model[-1].__class__.__name__
        # if detection_head_name == 'Detect_Yolo':
        #     from GEN1_od.models_spiking.loss_yolo import ComputeLoss_Yolo
        #     self.compute_loss = ComputeLoss_Yolo(self.model, self.cfg_loss)
        #     print('Using Yolo loss')
        # elif detection_head_name == 'Detect_Yolox':
        #     from GEN1_od.models_spiking.loss_yolox import ComputeLoss_Yolox
        #     self.compute_loss = ComputeLoss_Yolox(self.model, self.cfg_loss)
        #     print('Using Yolox loss')


        self.val_stats = None


    def forward(self, events_list) -> Tensor:
        return self.model(events_list)





    def validation_step(self, batch, batch_idx):
        events_list, label, event_path= batch
        # get the basename of the event file
        self.event_name = event_path[0].split('/')[-1]


        events_list = parse_event_data(events_list)
        targets, valid_mask = parse_label_data_yolo_infer(label)  # output label:[time_index, class, n_cx, n_cy, n_w, n_h]
        targets = filter_targets(targets, valid_mask)
        warm_up = 5

        gt_index = targets[:, 0].unique()
        # convert gt_index to list
        gt_index = gt_index.tolist()

        save_clean_Flag = False

        for time_idx, events in enumerate(events_list):
            # print(f'Processing time index {time_idx}')
            output = self.model.forward_backbone(events, time_idx)

            if time_idx >= warm_up -1 and time_idx in gt_index:
                preds, outputs = self.model.forward_detect(output) # pred (cx,cy, w,h, obj_conf, class...)
                bs, img_h, img_w = len(events_list[0]), 240, 304
                # after postprocess, yolox: (xl, yl, xr, yr, obj_conf, class_conf, class)
                preds = postprocess(preds, self.nc, conf_thre=0.03, nms_thre=0.5, class_agnostic=False)
                preds = filter_preds(preds, min_box_diag=30, min_box_side=10)


                # save the preds
                save_preds = True
                if save_preds:
                    import numpy as np
                    from utils.common import mkdir
                    import os
                    save_dir = 'GEN1_od/video_infer/' + self.config_name
                    mkdir(save_dir)
                    preds_save_path = os.path.join(save_dir, f'{batch_idx}_preds.txt')
                    gt_save_path = os.path.join(save_dir, f'{batch_idx}_gt.txt')
                    if not save_clean_Flag:
                        # if preds_save_path exist, delete it
                        if os.path.exists(preds_save_path):
                            os.remove(preds_save_path)
                        if os.path.exists(gt_save_path):
                            os.remove(gt_save_path)
                        save_clean_Flag = True

                    targets_temp = targets[targets[:, 0] == time_idx] # (time_index, class, n_cx, n_cy, n_w, n_h)
                    if targets_temp.shape[0] > 0: # gt exists
                        # preds_write (time_index, xl, yl, xr, yr, obj_conf, class_conf, class)
                        if preds[0] is not None:
                            preds_write = torch.cat( (torch.ones(preds[0].shape[0], 1, device=targets.device)*time_idx,preds[0]), dim=-1)
                            with open(preds_save_path, 'a') as f:
                                np.savetxt(f, preds_write.cpu().numpy(), fmt='%.6f')
                        else:
                            preds_write=np.zeros((1,8))
                            preds_write[0,0] = time_idx
                            with open(preds_save_path, 'a') as f:
                                np.savetxt(f, preds_write, fmt='%.6f')


                        gt_write = targets_temp
                        with open(gt_save_path, 'a') as f:
                            np.savetxt(f, gt_write.cpu().numpy(), fmt='%.6f')




                        # iouv = torch.linspace(0.5, 0.95, 10, device=targets.device)
                        # targets_temp[:,0]=0;
                        # self.val_stats = run_batch_metrics(preds, targets_temp, iouv, self.val_stats, img_h, img_w)
                        # pass


                plot = False
                if plot:
                    import numpy as np
                    from utils.psee_toolbox.visualize import vis_utils as vis
                    import cv2
                    # only plot one
                    bi = 0  # the batch index
                    # convert events to image
                    events_temp = events[0]
                    if events_temp.shape[0] > 0:
                        events_np = events_temp.cpu().numpy()
                        dtype = [('t', '<u4'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')]
                        events_np_structured = np.empty(events_np.shape[0], dtype=dtype)
                        events_np_structured['t'] = events_np[:, 0]
                        events_np_structured['x'] = events_np[:, 1]
                        events_np_structured['y'] = events_np[:, 2]
                        events_np_structured['p'] = events_np[:, 3]
                        img = vis.make_binary_histo(events_np_structured, img=None)
                    else:
                        img = np.zeros((img_h, img_w), dtype=np.uint8)

                    targets_temp = targets[targets[:, 0] == time_idx]
                    if targets_temp.shape[0] > 0: # targets_temp (time_index, class, n_cx, n_cy, n_w, n_h)
                        # plot the ground truth
                        for indx in range(targets_temp.shape[0]):  # plot the ground truth
                            cx_n, cy_n, w_n, h_n = targets_temp[indx][2:]
                            (cx, cy, w, h) = (cx_n * img_w, cy_n * img_h, w_n * img_w, h_n * img_h)
                            pt1 = (int(cx - w / 2), int(cy - h / 2))
                            pt2 = (int(cx + w / 2), int(cy + h / 2))
                            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)  # green

                    current_preds = preds[0]
                    if current_preds is not None: # preds(xl, yl, xr, yr, obj_conf, class_conf, class)
                        for indx in range(current_preds.shape[0]):
                            xl, yl, xr, yr = current_preds[indx][0], current_preds[indx][1], current_preds[indx][2], \
                                current_preds[indx][3]
                            pt1 = (int(xl), int(yl))
                            pt2 = (int(xr), int(yr))
                            obj_conf = current_preds[indx][4].item()
                            obj_conf_str = "{:.2f}".format(obj_conf)
                            class_ind = int(current_preds[indx][6])
                            if class_ind == 0: # car
                                color = tuple(reversed((255, 69, 0))) # red
                            if class_ind == 1: # pedestrian
                                color = tuple(reversed((255, 127, 36))) # yellow


                            class_name = self.names[class_ind]
                            cv2.rectangle(img, pt1, pt2, color, 2)
                            cv2.putText(img, obj_conf_str, (int(pt2[0] - 10), int(pt2[1] + 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(img, class_name, (int(pt1[0]), int(pt1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    cv2.imshow('out', img)
                    cv2.waitKey(0)

                    save_video_Flag = True
                    if save_video_Flag:
                        self.save_video.append(img)
                    pass



        functional.reset_net(self.model)

        # bs, img_h, img_w = len(events_list[0]), 240, 304
        # # after postprocess, yolox: (xl, yl, xr, yr, obj_conf, class_conf, class)
        # preds = postprocess(preds, self.nc, conf_thre=0.4, nms_thre=0.5, class_agnostic=False)
        # targets = filter_targets(targets, valid_mask)
        # preds = filter_preds(preds, min_box_diag=30, min_box_side=10)
        #
        # iouv = torch.linspace(0.5, 0.95, 10, device=targets.device)  # iou vector for mAP@0.5:0.95
        # self.val_stats = run_batch_metrics(preds, targets, iouv, self.val_stats, img_h, img_w)
        #







    def on_validation_start(self) -> None:
        # self.compute_loss.__init__(self.model, self.cfg_loss)
        pass



    def on_validation_epoch_start(self) -> None:
        self.val_stats = []
        self.save_video = []
        self.event_name = None


    def on_validation_epoch_end(self) -> None:
        if len(self.save_video)>0:
            import cv2
            import os
            # save_path = os.path.join('GEN1_od/video_infer/', self.config_name, self.event_name + '.avi')
            save_path = self.event_name + '.avi'
            videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), 200, (304, 240))
            for image in self.save_video:
                videoWriter.write(image)
            videoWriter.release()
            pass



        # map50, map = compute_metrics(self.val_stats, names=self.names)
        # self.log('map50', map50, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log('map', map, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #


def filter_targets(targets, valid_mask):
    targets = targets[(valid_mask[:,1]).bool(),:]
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


# parse the label for yolo detection
def parse_label_data_yolo_infer(label, image_w = 304, image_h = 240):
    from utils.common import xyxy2xywhn
    bboxes = label['bboxes']
    index = label['index']
    labels = label['labels']
    valid = label['valid']

    temp_box = xyxy2xywhn(bboxes, image_w, image_h, clip=False)

    out_targets = torch.cat((index.unsqueeze(-1), labels.unsqueeze(-1), temp_box), dim=-1) # [time_index, class, n_cx, n_cy, n_w, n_h]
    out_targets = out_targets.squeeze(0)
    valid_mask = torch.cat((index.unsqueeze(-1), valid.unsqueeze(-1)), dim=-1)
    valid_mask = valid_mask.squeeze(0)
    return out_targets, valid_mask