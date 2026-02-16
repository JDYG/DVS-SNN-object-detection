
import numpy as np

from utils.custom_collate_fn import collate_keep_dict

from utils.transform import Compose, RandomCrop, RandomFlip, RandomResize, Padding, Resize, ResizeInside

from lightning.pytorch.strategies import DDPStrategy
import datetime

'''
o: one stride

o o o o o|o o o o o|o o o o o||...|o o o o o|o o o o o|o o o o o||
-------->|-------->|-------->||...|-------->|-------->|-------->||  forward
                   |<--------||...                    |<--------||  backward

|--move--|--move--|      here, each move has 5 strides     
|----------seg dur-----------| ...|-------------seg dur---------|
|-----------------------------train duration--------------------|   
'''


DELTA_T = 20e3
MOVE_DURATION = DELTA_T * 5
SEG_DURATION = 500e3 # should be divisible by 500e3
TRAIN_DURATION = SEG_DURATION * 5



INPUT_SIZE = (240, 304)
LATENT_SIZE = (60, 76) # the size of the latent space, 240/4, 304/4

#############################Training Params######################################################
BATCH_SIZE = 1
TRAINING_DEVICES = [1]
MAX_EPOCHS = 20
seed_number = 42


cfg_names = {0: 'car', 1: 'pedestrian'}
nc = len(cfg_names) # number of classes

cfg_segment = dict(
    seg_num = int(TRAIN_DURATION // SEG_DURATION),
    move_per_seg = int(SEG_DURATION // MOVE_DURATION),
    move_steps = int(MOVE_DURATION // DELTA_T),
)

# embedding and cross attention
cfg_embed = dict(
    type = 'EventEmbedding_NoPadding', #  if the input batch size > 1，the events in the same time bin will be padded to the same length
    embed_type = ['MLP', 'MLP', 'MLP'], # MLP/Positional/Gaussian
    activation = 'relu',
    input_size    = INPUT_SIZE,
    latent_size = LATENT_SIZE,
    dynamic_dim   = [16, 16, 16],
    out_dim       = [16, 16, 16], # the first one is the dim for xy, the second one is the dim for time, the third one is the dim for polarity
    duration      = DELTA_T,
    discrete_time = True,
    time_bins     = 100, # spilt the duration into 100 bins
    shift_normalize = [True, True, False],# xy, t, p
    scale_normalize = [True, True, False], # xy, t, p
    max_log_scale = [4.0, 10.0, 2.0],
    xy_relative = True, # if True, using the relative position in a window for xy; if False, using the xy position in the whole frame
    time_steps = int(TRAIN_DURATION // DELTA_T),
)

cfg_attention = dict(
    type = 'SparseOneWindowAttention_NoPadding',
    embed_out_dim = sum(cfg_embed['out_dim']),
    latent_dim = 16, # dim for each head
    num_heads = 4,
    out_dim = 64,
    # hier_memory_type = 'PLIF', # PLIF/LIF, hierarchical memory
    # hier_tau = [1.0 / (1- np.exp( np.config_yoloxlog(0.5)/10 )), 1.0 / (1- np.exp( np.log(0.5)/30 )), 1.0 / (1- np.exp( np.log(0.5)/50 ))],
    # spiking neurons with three different decay time

)

cfg_latent_memory = dict(
    type = 'Latent_Memory',
    latent_neuron_type = 'PLIF', # PLIF/LIF/Mp_PLIF/Mp_LIF/LIF_soft/PLIF_soft
    latent_tau = 1.0 / (1- np.exp( np.log(0.5)/2.5 )),
    latent_size = LATENT_SIZE, # the size of the latent space, 240/4, 304/4
    latent_dim = cfg_attention['out_dim'], # dim for each head
)


tau1 = 1.0 / (1 - np.exp(np.log(0.5) / 2.5)) # fast decay
# tau2 = 1.0 / (1 - np.exp(np.log(0.5) / 6))
# tau3 = 1.0 / (1 - np.exp(np.log(0.5) / 9))




cfg_Detection = dict(
    type = 'Detection',
    nc = nc,
    anchors = None,
    input_channel = cfg_latent_memory['latent_dim'],
    depth_multiple = 1.0,  # model depth multiple
    width_multiple = 1.0,  # layer channel multiple

    # neuron type: LIF, PLIF, Mp_LIF
    # bn_type: BN, tdBN, TEBN, None
    #------------from, number, module, args


    backbone=[
        [-1, 1, 'Conv_BN', [128, 1, 1, None, 1, 'BN']], # 1 (60,76,128)
        [-1, 1, 'LCB', [128, 3, 1, None, 1, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # 2, (60,76,128)
        [-1, 1, 'C3_LCB', [128, True, 0.5, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # [⭐] 3, (60,76,128)
        [-1, 1, 'LCB', [256, 3, 2, None, 1, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # 4, (30,38,256)
        [-1, 1, 'C3_LCB', [256, True, 0.5, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # [⭐] 5, (30,38,256)
        [-1, 1, 'LCB', [512, 3, 2, None, 1, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # 6, (15,19,512)
        [-1, 1, 'C3_LCB', [512, True, 0.5, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # 7, (15,19,512)
    ],

    head=[
        [-1, 1, 'LCB', [256, 1, 1, None, 1, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # [⭐] 8, (15,19,256)
        [-1, 1, 'UpSample', ['nearest', 2]],  # 9  (30,38,256)
        [[-1, 5], 1, 'Concat', [1]],  # 10 (30,38, 256+256)
        [-1, 1, 'C3_LCB', [256, True, 0.5, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # 11, (30,38, 256)
        [-1, 1, 'LCB', [128, 1, 1, None, 1, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # [⭐] 12, (30,38, 128)
        [-1, 1, 'UpSample', ['nearest', 2]],  # 13 (60,76, 128)
        [[-1, 3], 1, 'Concat', [1]],  # 14 (60,76, 128+128)
        [-1, 1, 'C3_LCB', [128, True, 0.5, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # 15 (60,76, 128)
        [-1, 1, 'Mp_layer', [tau1, 'Mp_PLIF', 'BN']],  # 16

        [-1, 1, 'LCB', [128, 3, 2, None, 1, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # 17, (30,38, 128)
        [[-1, 12], 1, 'Concat', [1]],  # 18 (30,38, 128+128)
        [-1, 1, 'C3_LCB', [256, True, 0.5, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # 19 (30,38, 256)
        [-1, 1, 'Mp_layer', [tau1, 'Mp_PLIF', 'BN']],  # 20

        [-1, 1, 'LCB', [256, 3, 2, None, 1, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # 21 (15,19,256)
        [[-1, 8], 1, 'Concat', [1]],  # 22 (15,19, 256+256)
        [-1, 1, 'C3_LCB', [512, True, 0.5, 'PLIF', tau1, 'BN', 's', cfg_embed['time_steps']]],  # 23 (15,19,512)
        [-1, 1, 'Mp_layer', [tau1, 'Mp_PLIF', 'BN']],  # 24

        [[16, 20, 24], 1, 'Detect_YoloCS3', [nc, 128,0]],  # 25
    ],

)

cfg_pretrain = dict(


)

nl = 3
cfg_loss = dict(
    # box = 0.05 * 3 / nl, # box loss gain
    # cls = 0.3 * nc /80 * 3 / nl, # cls loss gain
    # obj = 0.7 * ( 272 / 640) **2 * 3 / nl, # obj loss gain (scale with pixels)
    # label_smoothing = 0.0,
    # cls_pw = 1.0, # cls BCELoss positive_weight
    # obj_pw = 1.0, # obj BCELoss positive_weight
    # iou_t = 0.20, # IoU training threshold
    # anchor_t = 4.0, # anchor-multiple threshold
    fl_gamma = 0.0, # focal loss gamma (efficientDet default gamma=1.5)
)



cfg_freeze_permanent = [
    # ['embedding.'],
    # ['attention.'],
    # ['latent_mem.'],

]

cfg_freeze_warm_up = [
    # ['.weight'],
]


init_lr = 5e-6
# final_lr = 0.00005
# max_lr = 0.0006
cfg_optimizer = dict(
    optimizer = 'Adam',
    lr0 = init_lr,
    weight_decay = 1e-4,
    momentum = 0.937,

    warm_up_steps=100,
    warm_up_epoches=3,
    warm_up_momentum=0.8,
    warm_up_bias_lr = 0.01,
    warm_up_tau_lr = 0.01,

)
cfg_lr = dict(
    # max_lr = max_lr,
    # pct_start=0.05,
    # steps_per_epoch = 1,
    epochs = MAX_EPOCHS,
    # div_factor = max_lr / init_lr,
    # final_div_factor = init_lr / final_lr,

)




cfg_pl_training = dict(
    accelerator='gpu',
    devices=TRAINING_DEVICES,
    max_epochs=MAX_EPOCHS,
    # strategy='ddp',
    # strategy=DDPStrategy(find_unused_parameters=True,timeout=datetime.timedelta(seconds=7200000)),
)


##===========Dataset Settings================

train_transform = Compose([
# RandomResize(scale_min=0.9, scale_range=2, event_downsampling='NONE', event_upsampling='NONE', event_resampling='NONE'), # RandomResize 会使得图像的大小发生变化
# Padding(size=INPUT_SIZE, halign='center', valign='center', const_image=0, const_mask=-1, padding_mode='constant'), # padding的效果是，是的resize后的图像居中，因为如果resize后大小小于原来的camera大小(240,304)， 那么需要将图像放在中间
# RandomCrop(crop_size=INPUT_SIZE, const_image=0, clip_border=False, bbox_filter_by_center=True), # 随机crop图像，使得图像大小为(240,304)，dlip_border=False表示bounding box允许到图像边界外面，否则会被裁剪到[0,w]和[0,h]. bbox_filter_by_center=True表示只bbox保留中心点在图像内部的bounding box
RandomFlip(prob=0.5, direction='H'),])

# train_transform = None

cfg_TrainingData = dict(
    fpath_evt_lst='./GEN1_od/data/list/train/events.txt',
    fpath_lbl_lst='./GEN1_od/data/list/train/labels.txt',
    train_duration=TRAIN_DURATION,
    stride_t=DELTA_T,
    move_duration=MOVE_DURATION,
    seg_duration=SEG_DURATION,
    event_transform=train_transform,
    high_light_p=None,

    loader_parm = dict(
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True, #??
        drop_last=True,
        collate_fn=collate_keep_dict,
    ),
)

cfg_ValidationData = dict(
    fpath_evt_lst='./GEN1_od/data/list/val/events.txt',
    fpath_lbl_lst='./GEN1_od/data/list/val/labels.txt',
    train_duration=TRAIN_DURATION,
    stride_t=DELTA_T,
    move_duration=MOVE_DURATION,
    seg_duration=SEG_DURATION,
    event_transform=None,
    high_light_p=None,


    loader_parm = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True, #??
        drop_last=True,
        collate_fn=collate_keep_dict,
    ),
)






pretrained_model = None
exec_string="""

"""