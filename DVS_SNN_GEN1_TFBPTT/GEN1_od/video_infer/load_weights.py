import torch
import matplotlib.pyplot as plt


pretrained_path = 'GEN1_od/experiments/checkpoints/config_20e3_hard/epoch=06-step=30765-val_loss=3.59.ckpt'
state_dict = torch.load(pretrained_path, map_location='cuda:0')['state_dict']

for name, param in state_dict.items():
    # print(name)
    if 'model.embedding.' in name:
        pass
    elif 'model.attention.' in name:
        pass
    elif 'model.latent_mem.' in name:
        pass
    elif 'model.detection.' in name:
        name_ = name.split('model.detection.')[1]
        layer = int(name_.split('.')[1])

        if layer >= 24: # we only pay attention to layer 0 ~ layer 23
            break

        if 'conv.weight' in name_:
            weight_temp = param.cpu().numpy()
            min_, max_ = weight_temp.min(), weight_temp.max()
            print(name,  f'\t min: {min_}, max: {max_}')

            # plot the distributation of conv weights
            plt.figure()
            plt.hist(weight_temp.flatten(), bins = 100)
            plt.title(name)

plt.show()











