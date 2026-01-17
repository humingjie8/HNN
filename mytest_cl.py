import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from PIL import Image
from lib.PraNet_Res2Net_cl import PraNet
# from utils.dataloader import test_dataset
from utils.dataloader_cl_v2 import get_loaders
import cv2  # OpenCV is required
from config_test import get_parser

"""
Evaluation / inference script for PraNet in continual-learning setup.

This script loads model checkpoints and runs inference on datasets
organized per-task. It saves predicted saliency maps, ground-truth
masks, original images and overlay visualizations. The script supports
multi-head models (per-task heads) produced by the continual-learning
adaptations in this repository.
"""

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    image_root = r'D:\PolypGen2021_MultiCenterData_v3'
    gt_root = r'D:\PolypGen2021_MultiCenterData_v3' 
    data_loaders = get_loaders(image_root, batchsize=32, trainsize=256,ratio_set=[1.0,0.0,0.0])
    task_num = len(data_loaders)


    # Get all weight files in the directory
    weight_files = [os.path.join(opt.weights_dir, f)
                for f in os.listdir(opt.weights_dir)
                if f.startswith('PraNet-') and f.endswith('.pth')]
    print(weight_files)
    # for weight_file in weight_files:

    for weight_index, weight_file in enumerate(weight_files):
        opt.pth_path = weight_file  # Update opt.pth_path to the current weight file
        model = PraNet(task_num=task_num)
        # model = PraNet()
        model.load_state_dict(torch.load(weight_file))
        model.cuda()
        model.eval()

        print('process on:', opt.pth_path)

        # Extract the base name of the weight file without extension
        model_name = os.path.splitext(os.path.basename(opt.pth_path))[0]

        # for x, loaders in enumerate(data_loaders):
        for t, loaders in enumerate(data_loaders):
            # if t > weight_index:
            #     break  # Skip loaders beyond the current weight file's range
            name = loaders['name']
            train_loader = loaders['train_loader']
            val_loader = loaders['val_loader']
            test_loader = loaders['test_loader']

            # Create the directory based on the model name, under weights_dir
            save_path = os.path.join(opt.weights_dir, f'results/{opt.approches}/{model_name}/{name}/')
            os.makedirs(save_path, exist_ok=True)


            os.makedirs(os.path.join(save_path, 'outputs'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'masks'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'overlay'), exist_ok=True)


            global_counter = 0  # initialize global counter

            for i, pack in enumerate(train_loader, start=1):
                images, gts = pack
                
                for j in range(len(images)):
                    global_counter += 1  # increment global counter
                    image = images[j]

                    gt = gts[j]
                    gt = np.asarray(gt, np.float32)
                    gt /= (gt.max() + 1e-8)

                    # add batch dimension for the image tensor
                    image = image.unsqueeze(0).cuda()
                    gt = np.expand_dims(gt, axis=0)

                    image = image.cuda()

                    # forward pass through model; model returns multi-scale maps
                    res5, res4, res3, res2 = model(image)
                    res = res2[t]
                    # res = res2   # single-head variant (not used here)
                    
                    # no need to expand gt batch dimension for sizing beyond above
                    res = F.interpolate(res, size=gt.shape[2:], mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                    # convert to PIL image and save
                    res_img = Image.fromarray((res * 255).astype(np.uint8))
                    save_filename = os.path.join(save_path, 'outputs', f'{global_counter}.png')
                    res_img.save(save_filename)

                    # save original gt mask
                    gt_img = Image.fromarray((gt.squeeze() * 255).astype(np.uint8))
                    gt_save_filename = os.path.join(save_path,'masks', f'{global_counter}.png')
                    gt_img.save(gt_save_filename)

                    # undo preprocessing and save original image
                    image_np = image.squeeze().cpu().detach()  # remove batch dim and move to CPU
                    image_np = denormalize(image_np, mean, std).permute(1, 2, 0).numpy()  # unnormalize and reorder
                    image_np = (image_np * 255).astype(np.uint8)  # restore pixel values
                    image_np = Image.fromarray(image_np, mode='RGB')  # make PIL RGB image
                    image_pil = image_np.convert('RGBA')  # convert to RGBA to add alpha channel

                    # set overlay alpha value
                    alpha_value = 128  # alpha (0-255); adjust as desired
                    alpha_channel = Image.new('L', image_pil.size, alpha_value)
                    image_pil.putalpha(alpha_channel)

                    image_np.save(os.path.join(save_path, 'images', f'{global_counter}.png'))
    
                    # apply color map using OpenCV and convert BGR->RGB
                    res_colormap = cv2.applyColorMap((res * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    res_colormap = cv2.cvtColor(res_colormap, cv2.COLOR_BGR2RGB)


                    # convert color-mapped result to PIL RGBA image
                    res_pil = Image.fromarray(res_colormap)
                    res_pil = res_pil.convert('RGBA')


                    # create an alpha channel from the saliency map values
                    alpha_data = (res.squeeze() * 255).astype(np.uint8)
                    alpha_channel = Image.fromarray(alpha_data,mode='L')
                    res_pil.putalpha(alpha_channel)


                    # ensure both images have the same size
                    if image_pil.size != res_pil.size:
                        res_pil = res_pil.resize(image_pil.size)

                    # composite overlay and save
                    overlay_img = Image.alpha_composite(res_pil, image_pil)

                    # overlay_img_pil = Image.fromarray(overlay_img)
                    overlay_save_filename = os.path.join(save_path, 'overlay', f'{global_counter}.png')
                    overlay_img.save(overlay_save_filename)