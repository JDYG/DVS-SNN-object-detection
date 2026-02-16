# using the whole video for test

import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_yolox_20e3_5_64attn.py', help='Config file')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
# parser.add_argument('--bs', type=int, default=2, help='batch size')
args = parser.parse_args()


from importlib import machinery
from torch.utils.data import DataLoader, Subset
import lightning as L


# from GEN1_od.video_infer.module_infer_compute_gt import Gen1DetectionModule
from GEN1_od.video_infer.module_infer_show_mp import Gen1DetectionModule


import random
import numpy as np
from utils.common import get_list
from GEN1_od.video_infer.load_video import VideoLoader


class Gen1DataModule(L.LightningDataModule):
    def __init__(self, cfg_TrainingData):
        super().__init__()
        self.cfg_TrainingData = cfg_TrainingData

    def setup(self, stage: str):
        if stage == 'validate':
            test_evt_lst = './GEN1_od/data/list/test/events.txt'
            test_lbl_lst = './GEN1_od/data/list/test/labels.txt'

            # or test on one video
            test_evt_lst = '/Gen1_Automotive/train/17-03-30_12-53-58_122500000_182500000_td.dat'
            test_lbl_lst = None


            self.validataion_dataset = VideoLoader(test_evt_lst, test_lbl_lst, stride_t=self.cfg_TrainingData['stride_t'])



    def val_dataloader(self):
        return DataLoader(self.validataion_dataset, shuffle=False, batch_size=1)



def main():
    try:
        config_path = f"GEN1_od/config_test/{args.config}"
        print(f"Using config file: {config_path}")
        config_module = machinery.SourceFileLoader('config', config_path).load_module()
    except ImportError:
        print("Error: Config file not found or invalid")
        return
    if args.gpu is not None:
        config_module.cfg_pl_training['devices'] = [args.gpu]





    cfg = dict(cfg_names=config_module.cfg_names,
               cfg_embed=config_module.cfg_embed,
               cfg_attention=config_module.cfg_attention,
               cfg_latent_memory=config_module.cfg_latent_memory,
               cfg_Detection=config_module.cfg_Detection,
               cfg_pretrain=config_module.cfg_pretrain,
               cfg_loss=config_module.cfg_loss,
               cfg_optimizer=config_module.cfg_optimizer,
               cft_lr=config_module.cfg_lr,
               cfg_freeze_permanent=config_module.cfg_freeze_permanent,
               cfg_freeze_warm_up=config_module.cfg_freeze_warm_up,
               exec_string=config_module.exec_string,
               config_name = args.config[:-3]
               )

    cfg_training_data = config_module.cfg_TrainingData
    data_module = Gen1DataModule(cfg_training_data)

    module = Gen1DetectionModule(cfg)

    trainer = L.Trainer(**config_module.cfg_pl_training, logger=False, callbacks=[],
                        gradient_clip_val=None, gradient_clip_algorithm='norm', num_sanity_val_steps=0,

                        precision='16-mixed',
                        enable_progress_bar=True,
                        detect_anomaly=False,
                        enable_checkpointing=False)

    config_module.pretrained_model = './GEN1_od/experiments/checkpoints/model.ckpt'
    print(f"Loading pretrained model: {config_module.pretrained_model}")
    trainer.validate(module, data_module, ckpt_path=config_module.pretrained_model)





if __name__ == '__main__':
    main()

