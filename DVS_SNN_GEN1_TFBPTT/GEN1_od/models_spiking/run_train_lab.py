import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_yolox_5e3_10_PLIF_all_BN.py', help='Config file')
parser.add_argument('--gpu', type=int, help='gpu index')
parser.add_argument('--bs', type=int, help='batch size')
args = parser.parse_args()



from importlib import machinery
from torch.utils.data import DataLoader, Subset
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from GEN1_od.models_spiking.module_test import Gen1DetectionModule
from GEN1_od.models_spiking.gen1_data_packet import EventPacketStream
import random
import numpy as np


class Gen1DataModule(L.LightningDataModule):
    def __init__(self, cfg_TrainingData, cfg_ValidationData=None):
        super().__init__()
        self.cfg_TrainingData = cfg_TrainingData
        self.cfg_ValidationData = cfg_ValidationData



    def setup(self, stage: str):

        if stage == 'fit':
            self.train_dataset = EventPacketStream(fpath_evt_lst = self.cfg_TrainingData['fpath_evt_lst'],
                                                   fpath_lbl_lst = self.cfg_TrainingData['fpath_lbl_lst'],
                                                   train_duration = self.cfg_TrainingData['train_duration'],
                                                   stride_t = self.cfg_TrainingData['stride_t'],
                                                   warm_up=self.cfg_TrainingData['warm_up'],
                                                   event_transform = self.cfg_TrainingData['event_transform'],
                                                   hight_light_p=self.cfg_TrainingData['high_light_p'])
            random_indices = random.sample(range(len(self.train_dataset)), len(self.train_dataset))
            # l = int(len(self.train_dataset) * 1.0)
            l = 20*6

            random_indices = random_indices[0:l]


            self.train_subset = Subset(self.train_dataset, random_indices)
            self.val_subset = Subset(self.train_dataset, random_indices)


            # # validation
            # self.validataion_dataset = EventPacketStream(fpath_evt_lst = self.cfg_ValidationData['fpath_evt_lst'],
            #                                        fpath_lbl_lst = self.cfg_ValidationData['fpath_lbl_lst'],
            #                                        train_duration = self.cfg_ValidationData['train_duration'],
            #                                        stride_t = self.cfg_ValidationData['stride_t'],
            #                                        event_transform = self.cfg_ValidationData['event_transform'],
            #                                        )
            #
            # random_indices = random.sample(range(len(self.validataion_dataset)), len(self.validataion_dataset))
            # # l = int(len(self.validataion_dataset) * 0.5)
            # l = 20*6
            # random_indices = random_indices[0:l]
            # self.val_subset = Subset(self.validataion_dataset, random_indices)


            print('\n')
            print(self.train_subset[0][3])
            print(self.val_subset[0][3])

        if stage == 'validate':
            self.validataion_dataset = EventPacketStream(fpath_evt_lst=self.cfg_ValidationData['fpath_evt_lst'],
                                                         fpath_lbl_lst=self.cfg_ValidationData['fpath_lbl_lst'],
                                                         train_duration=self.cfg_ValidationData['train_duration'],
                                                         stride_t=self.cfg_ValidationData['stride_t'],
                                                         warm_up=self.cfg_ValidationData['warm_up'],
                                                         event_transform=self.cfg_ValidationData['event_transform'],
                                                         )

            random_indices = random.sample(range(len(self.validataion_dataset)), len(self.validataion_dataset))
            l = int(len(self.validataion_dataset) * 0.01)
            random_indices = random_indices[0:l]
            self.val_subset = Subset(self.validataion_dataset, random_indices)

            print('\n')
            print(self.val_subset[0][3])



    def train_dataloader(self):
        return DataLoader(self.train_subset, **self.cfg_TrainingData['loader_parm'])
    def val_dataloader(self):
        val_loader_parm = self.cfg_ValidationData['loader_parm']
        val_loader_parm['shuffle'] = False
        # val_loader_parm['batch_size'] = 8
        return DataLoader(self.val_subset, **val_loader_parm)





def main(args):
    torch.set_float32_matmul_precision('medium')
    # Dynamically import config file
    try:
        config_path = f"GEN1_od/config_test/{args.config}"
        print(f"Using config file: {config_path}")
        config_module = machinery.SourceFileLoader('config', config_path).load_module()
    except ImportError:
        print("Error: Config file not found or invalid")
        return

    if args.gpu is not None:
        config_module.cfg_pl_training['devices'] = [args.gpu]
    if args.bs is not None:
        config_module.cfg_TrainingData['loader_parm']['batch_size'] = args.bs
        config_module.cfg_ValidationData['loader_parm']['batch_size'] = args.bs

    # if config_module has seed_number
    if hasattr(config_module, 'seed_number'):
        seed_number = config_module.seed_number
    else:
        seed_number = 42
    print(f"Setting seed to {seed_number}")
    L.seed_everything(seed_number)

    cfg = dict(cfg_names=config_module.cfg_names,
               cfg_embed=config_module.cfg_embed,
               cfg_attention=config_module.cfg_attention,
               cfg_latent_memory=config_module.cfg_latent_memory,
               cfg_Detection=config_module.cfg_Detection,
               cfg_pretrain = config_module.cfg_pretrain,
               cfg_loss=config_module.cfg_loss,
               cfg_optimizer=config_module.cfg_optimizer,
               cft_lr = config_module.cfg_lr,
               cfg_freeze_permanent=config_module.cfg_freeze_permanent,
               cfg_freeze_warm_up=config_module.cfg_freeze_warm_up,
               exec_string = config_module.exec_string,
               cfg_segment=config_module.cfg_segment,
               )

    cfg_training_data = config_module.cfg_TrainingData
    cfg_validation_data = config_module.cfg_ValidationData
    data_module = Gen1DataModule(cfg_training_data, cfg_validation_data)



    module = Gen1DetectionModule(cfg)

    version_name = args.config.split('.')[0]
    checkpoint_callback = ModelCheckpoint(dirpath='./GEN1_od/experiments/checkpoints/'+version_name,
                                          filename='{epoch:02d}-{step:04d}-{val_loss:.2f}', auto_insert_metric_name=True,
                                          save_weights_only=False, save_top_k=-1,
                                          # every_n_train_steps=3000,
                                          every_n_epochs=20,
                                          save_last=True)
    wandb_logger = WandbLogger(project='DVS_SNN_4', name=version_name, save_dir='./GEN1_od/experiments/', )
    # wandb_logger.watch(module.model, log='all', log_freq=1000)
    trainer = L.Trainer(**config_module.cfg_pl_training, logger=wandb_logger, callbacks=[checkpoint_callback],
                        gradient_clip_val=None, gradient_clip_algorithm='norm', num_sanity_val_steps=0,
                        # limit_train_batches=300,
                        # limit_val_batches=300,
                        # overfit_batches=100,
                        precision='16-mixed',
                        enable_progress_bar=True,
                        detect_anomaly=False,
                        enable_checkpointing=True)
    # trainer.fit(module, data_module)


    if config_module.pretrained_model is not None:
        print(f"Loading pretrained model: {config_module.pretrained_model}")
        trainer.fit(module, data_module, ckpt_path=config_module.pretrained_model)
    else:
        trainer.fit(module, data_module)



if __name__ == "__main__":
    main(args)