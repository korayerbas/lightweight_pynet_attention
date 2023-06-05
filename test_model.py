# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import numpy as np
import sys
import os

from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from load_data import LoadVisualData
from model_attention import PyNET_att
import utils
import imageio

to_image = transforms.Compose([transforms.ToPILImage()])

level, restore_epoch, dataset_dir, use_gpu, orig_model = utils.process_test_model_args(sys.argv)
#level, restore_epoch, dataset_dir, use_gpu, orig_model =  0, 57, '/content/gdrive/MyDrive/ColabNotebooks/pynet_fullres_dataset', "true", "true"
dslr_scale = float(1) / (2 ** (level - 1))


def test_model():

    if use_gpu == "true":
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = torch.device("cpu")

    # Creating dataset loaders

    visual_dataset = LoadVisualData(dataset_dir, 8, dslr_scale, level, full_resolution=True)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)

    # Creating and loading pre-trained PyNET model

    model = PyNET_att(level=level, instance_norm=True, instance_norm_level_1=True).to(device)
    model = torch.nn.DataParallel(model)

    if orig_model == "true":
        model.load_state_dict(torch.load("/content/gdrive/MyDrive/ColabNotebooks/PYNET/models/original/pynet_level_0.pth"), strict=True)
    else:
        model.load_state_dict(torch.load("/content/gdrive/MyDrive/ColabNotebooks/pynet_fullres/model/pynet_level_0_epoch_59_lght.pth"), strict=True)

    if use_gpu == "true":
        model.half()

    model.eval()

    # Processing full-resolution RAW images

    with torch.no_grad():

        visual_iter = iter(visual_loader)
        for j in range(len(visual_loader)):

            print("Processing image " + str(j))

            torch.cuda.empty_cache()
            raw_image = next(visual_iter)

            if use_gpu == "true":
                raw_image = raw_image.to(device, dtype=torch.half)
            else:
                raw_image = raw_image.to(device)

            # Run inference

            enhanced = model(raw_image.detach())
            enhanced = np.asarray(to_image(torch.squeeze(enhanced.float().detach().cpu())))

            # Save the results as .png images

            if orig_model == "true":
                imageio.imwrite("/content/gdrive/MyDrive/ColabNotebooks/pynet_fullres/dataset/full_res_results/" + str(j) + "_level_" + str(level) + "_orig.png", enhanced)
            else:
                imageio.imwrite("/content/gdrive/MyDrive/ColabNotebooks/pynet_fullres/dataset/full_res_results/" + str(j) + "_level_" + str(level) +
                        "_epoch_" + str(restore_epoch) + "_lght.png", enhanced)


if __name__ == '__main__':
    test_model()
