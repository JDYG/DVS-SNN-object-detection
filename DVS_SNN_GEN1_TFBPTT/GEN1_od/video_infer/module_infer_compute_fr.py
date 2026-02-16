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

from visualizer import get_local
get_local.activate()

from utils.torch_utils import smart_optimizer
from utils.plots import plot_lr_scheduler
from utils.common import non_max_suppression

from GEN1_od.models_spiking.val import run_batch_metrics, compute_metrics
import time
import numpy as np
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







        self.val_stats = None


    def forward(self, events_list) -> Tensor:
        return self.model(events_list)





    def validation_step(self, batch, batch_idx):
        events_list, label, event_path= batch
        self.event_name = event_path[0].split('/')[-1]

        events_list = parse_event_data(events_list)




        targets, valid_mask = parse_label_data_yolo_infer(label)
        targets = filter_targets(targets, valid_mask)
        warm_up = 20

        gt_index = targets[:, 0].unique()
        gt_index = gt_index.tolist()

        save_clean_Flag = False

        for time_idx, events in enumerate(events_list):

            output = self.model.forward_backbone(events, time_idx)


            if True:
                preds, outputs = self.model.forward_detect(output)

                bs, img_h, img_w = len(events_list[0]), 240, 304
                preds = postprocess(preds, self.nc, conf_thre=0.2, nms_thre=0.5, class_agnostic=False)
                preds = filter_preds(preds, min_box_diag=30, min_box_side=10)




                plot = False
                if plot:
                    import numpy as np
                    from utils.psee_toolbox.visualize import vis_utils as vis
                    import cv2
                    bi = 0
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
                    if targets_temp.shape[0] > 0:
                        for indx in range(targets_temp.shape[0]):
                            cx_n, cy_n, w_n, h_n = targets_temp[indx][2:]
                            (cx, cy, w, h) = (cx_n * img_w, cy_n * img_h, w_n * img_w, h_n * img_h)
                            pt1 = (int(cx - w / 2), int(cy - h / 2))
                            pt2 = (int(cx + w / 2), int(cy + h / 2))
                            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)

                    current_preds = preds[0]
                    if current_preds is not None:
                        for indx in range(current_preds.shape[0]):
                            xl, yl, xr, yr = current_preds[indx][0], current_preds[indx][1], current_preds[indx][2], \
                                current_preds[indx][3]
                            pt1 = (int(xl), int(yl))
                            pt2 = (int(xr), int(yr))
                            obj_conf = current_preds[indx][4].item()
                            obj_conf_str = "{:.2f}".format(obj_conf)
                            class_ind = int(current_preds[indx][6])
                            if class_ind == 0:
                                color = tuple(reversed((255, 69, 0)))
                            if class_ind == 1:
                                color = tuple(reversed((255, 127, 36)))


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



        cache = get_local.cache

        import numpy as np
        LCB_fr = np.stack(cache['LCB.forward'])
        Latent_Memory_fr = np.stack(cache['Latent_Memory.forward'])
        len_LCB_fr_per_iter = len(LCB_fr)/(time_idx+1)
        LCB_fr = LCB_fr[int(len_LCB_fr_per_iter*warm_up):,:]
        len_Latent_Memory_fr_per_iter = len(Latent_Memory_fr)/(time_idx+1)
        Latent_Memory_fr = Latent_Memory_fr[int(len_Latent_Memory_fr_per_iter*warm_up):,:]

        fr = np.vstack((LCB_fr, Latent_Memory_fr))
        fr = fr.sum(axis=0)
        fr = fr[0]/fr[1]
        self.latent_memory_frs.append(fr)
        print(f'fr: {fr}')

        functional.reset_net(self.model)







    def on_validation_start(self) -> None:
        pass



    def on_validation_epoch_start(self) -> None:
        self.val_stats = []
        self.save_video = []
        self.event_name = None


        self.latent_memory_frs = []



    def on_validation_epoch_end(self) -> None:
        if len(self.save_video)>0:
            import cv2
            import os
            save_path = self.event_name + '.avi'
            videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), 200, (304, 240))
            for image in self.save_video:
                videoWriter.write(image)
            videoWriter.release()
            pass


        self.latent_memory_frs = np.array(self.latent_memory_frs)
        print(f'fr mean: {self.latent_memory_frs.mean()}')



def filter_targets(targets, valid_mask):
    targets = targets[(valid_mask[:,1]).bool(),:]
    return targets

def filter_preds(preds, min_box_diag=30, min_box_side=10):
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


def parse_label_data_yolo_infer(label, image_w = 304, image_h = 240):
    from utils.common import xyxy2xywhn
    bboxes = label['bboxes']
    index = label['index']
    labels = label['labels']
    valid = label['valid']

    temp_box = xyxy2xywhn(bboxes, image_w, image_h, clip=False)

    out_targets = torch.cat((index.unsqueeze(-1), labels.unsqueeze(-1), temp_box), dim=-1)
    out_targets = out_targets.squeeze(0)
    valid_mask = torch.cat((index.unsqueeze(-1), valid.unsqueeze(-1)), dim=-1)
    valid_mask = valid_mask.squeeze(0)
    return out_targets, valid_mask