import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.PraNet_Res2Net_cl import PraNet
from utils.dataloader_cl_v2 import get_loaders
from utils.utils import clip_gradient, adjust_lr, AvgMeter, freeze_model
import torch.nn.functional as F
from copy import deepcopy
import csv  
"""
Continual-learning training script with EWC-based hippocampal learning paradigm.

This training script implements a PraNet-based training pipeline adapted
for continual learning. The key innovation is an Elastic Weight
Consolidation (EWC) regularization that is framed as a hippocampal
learning paradigm: the method simulates a hippocampal-like device that
captures the order and importance of previously learned tasks. In this
design, task-specific reverse-attention heads are supported and the
EWC terms are computed and applied to retain important parameters while
allowing adaptation to new tasks.

Detailed notes on Fisher storage and cognitive_map_HNN aggregation implementation:

- Per-task Fisher estimates: after training each task the code computes
    diagonal approximations of the Fisher information (via
    `compute_fisher_information`) and stores them in the `fisher_dict`.
    The saved structure is later serialized to `fisher_dict.pth` inside
    the task snapshot folder.

- Hippocampal-style aggregation (`cognitive_map_HNN`): when training on a
    new task (t>0) the implementation builds an aggregate importance map
    called `cognitive_map_HNN` by iterating over previously stored Fisher
    tensors. Each past Fisher tensor is decayed according to its age
    (age = `t - key`) using `decay_tensor` (the code uses a `'linear'`
    decay by default). This decayed aggregation explicitly captures both
    the order (recency) and the importance (magnitude of Fisher values)
    of past tasks — mirroring an artificial hippocampal device that
    records episodic statistics with fading memory.

- Use in regularization: `cal_ewc_loss` consumes either the most
    recently computed Fisher (`fisher_information`) or the aggregated
    `cognitive_map_HNN` depending on the mode and task index; the EWC penalty is
    applied to shared backbone parameters while ignoring task-specific
    head layers so that new head(s) remain plastic.

- Design intent: this combination (per-task Fisher storage + decayed
    aggregation + selective EWC application) implements a practical
    hippocampal learning paradigm for continual segmentation: episodic
    captures are stored, importance and recency are preserved via decay,
    and consolidation is performed through EWC-style protection.

"""

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    # compute weighted binary cross-entropy loss
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # apply sigmoid to predictions
    pred = torch.sigmoid(pred)
    
    # compute weighted intersection
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    
    # compute weighted union
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    
    # compute weighted IoU loss
    wiou = 1 - (inter + 1) / (union - inter + 1)
    
    # return mean of weighted BCE and weighted IoU losses
    return (wbce + wiou).mean()

def compute_fisher_information(model, train_loader,optimizer,t):
    fisher_information = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    model.eval()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        # ---- forward ----
        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
        # ---- loss function ----
        loss5 = structure_loss(lateral_map_5, gts)
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2[t], gts)
        loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                print(name, 'is none')
            if param.grad is not None:
                fisher_information[name] += param.grad.data.pow(2)

    for name in fisher_information:
        fisher_information[name] /= len(train_loader)

    return fisher_information

def cal_ewc_loss(model, model_old, fisher_information, lambda_ewc):
    """
    Compute the EWC (Elastic Weight Consolidation) loss.

    Parameters:
    model: current model
    model_old: previous model (frozen)
    fisher_information: dictionary mapping parameter names to Fisher values
    lambda_ewc: coefficient for the EWC regularization term

    Returns:
    ewc_loss: computed EWC loss scalar
    """
    ewc_loss = 0

    # define layer names to ignore
    ignored_layers = {'ra2_conv1', 'ra2_conv2', 'ra2_conv3', 'ra2_conv4',
                      'ra4_conv1','ra4_conv2','ra4_conv3','ra4_conv4','ra4_conv5',
                      'ra3_conv1','ra3_conv2','ra3_conv3','ra3_conv4','last',
                      }

    for param_name, param in model.named_parameters():

         # check whether to ignore this layer
        if any(layer in param_name for layer in ignored_layers):
            continue

        if param_name in fisher_information:

            # parameter from the old model
            param_old = model_old.state_dict()[param_name]
            # value from the Fisher information matrix
            fisher_value = fisher_information[param_name]
            # EWC loss term
            # compute parameter difference
            param_diff = param - param_old
            # compute EWC loss term
            current_ewc_loss = torch.sum(fisher_value * param_diff.pow(2))
            
            # 检查当前的 EWC 损失是否为 NaN
            if torch.isnan(current_ewc_loss):
                # check for NaNs in fisher_value or param_diff (debugging)
                # print(f"NaN detected in EWC loss for parameter: {param_name}")
                # print(f"fisher_value: {fisher_value}")
                # print(f"param: {param}")
                # print(f"param_old: {param_old}")
                # print(f"param_diff: {param_diff}")

                # You may add handling here, e.g., skip this parameter or abort training
                continue

            ewc_loss += current_ewc_loss

    ewc_loss = (lambda_ewc) * ewc_loss
    return ewc_loss

def decay_tensor(time, tensor, method):
    """Decay tensor values to model memory fading across tasks.

    This helper scales a tensor of importance values (e.g., Fisher)
    according to a chosen method. It is used to simulate forgetting or
    attenuation of importance for older tasks when aggregating past
    Fisher estimates — a component of the implemented hippocampal
    learning paradigm that models order and decreasing influence over
    time.

    Parameters:
        time (float or int): exponent that controls decay strength.
        tensor (torch.Tensor): importance tensor to decay.
        method (str): 'linear' or 'exponential' supported.

    Returns:
        torch.Tensor: decayed tensor of the same shape.
    """
    # normalization
    max_val = tensor.max()
    min_val = tensor.min()
    
    if torch.isnan(max_val) or torch.isnan(min_val):
        print("NaN detected in max_val or min_val")
        print(f"max_val: {max_val}, min_val: {min_val}")
        
    if tensor.numel() == 1 :
        normalized_tensor = tensor
        print("Single element tensor. Normalized tensor set to the original tensor.")

    elif max_val == min_val and max_val == 0.0:
        normalized_tensor = torch.zeros_like(tensor)
        print("Warning: All elements in tensor are the same. Normalized tensor set to zero.")

    else:
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
    
    if torch.isnan(normalized_tensor).any():
        print("NaN detected in normalized_tensor")
        print(f"tensor: {tensor}")
        print(f"normalized_tensor: {normalized_tensor}")
    
    # compute weights
    if method == 'linear':
        weights = normalized_tensor
    elif method == 'exponential':
        weights = torch.exp(normalized_tensor)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    if torch.isnan(weights).any():
        print("NaN detected in weights")
        print(f"weights: {weights}")
    
    # compute decayed tensor
    decayed_tensor = tensor * (weights ** time)
    
    if torch.isnan(decayed_tensor).any():
        print("NaN detected in decayed_tensor")
        print(f"decayed_tensor: {decayed_tensor}")
        print(f"tensor: {tensor}")
        print(f"weights: {weights}")
        print(f"time: {time}")
    
    return decayed_tensor


def train(data_loaders, model, optimizer, epoch):
    """Train loop supporting continual (sequential) tasks with EWC, BP and HNN.

    This function runs multi-scale training per task, computes Fisher
    information after finishing each task, and saves Fisher estimates
    for later EWC regularization. The training loop supports different
    modes controlled by `opt.approches` including 'bp', 'ewc', and
    'HNN'.

    Emphasis on hippocampal learning paradigm:
    - After each task finishes, Fisher information is computed and
        stored as an importance memory (simulating hippocampal storage of
        order and importance).
    - When training new tasks, EWC penalties use those stored
        importance values to protect crucial parameters while allowing
        task-specific heads to adapt, reflecting the interaction between
        hippocampus-like episodic captures and cortical consolidation.
    """
    model.train()
    model_old = deepcopy(model)
    fisher_dict = dict()

    # 获取当前日期和时间
    now = datetime.now()

    # format datetime for folder naming
    folder_name_time = now.strftime("%Y%m%d_%H%M%S")
    save_path = 'snapshots_polyp_gen/{}/{}/{}/'.format(opt.train_save, opt.approches, folder_name_time)
    os.makedirs(save_path, exist_ok=True)

    # ---- save training parameters to txt alongside weights ----
    param_txt_path = os.path.join(save_path, 'train_params.txt')
    with open(param_txt_path, 'w', encoding='utf-8') as f:
        for k, v in vars(opt).items():
            f.write(f"{k}: {v}\n")
    print(f"Training parameters saved to {param_txt_path}")

    # ++++ Added: initialize loss save directory ++++
    loss_save_dir = os.path.join('snapshots_polyp_gen', opt.train_save, opt.approches, folder_name_time)
    os.makedirs(loss_save_dir, exist_ok=True)
    loss_file_path = os.path.join(loss_save_dir, 'loss_history.csv')

    for t, loaders in enumerate(data_loaders):
        train_loader = loaders['train_loader']
        val_loader = loaders['val_loader']
        test_loader = loaders['test_loader']
        # This training function aims to improve model robustness via multi-scale training,
        # while recording losses at the end of each epoch and periodically saving model snapshots.
        for i_epoch in range(1, epoch):
            adjust_lr(optimizer, opt.lr, i_epoch, opt.decay_rate, opt.decay_epoch)
            # train(data_loader, model, optimizer, epoch)
            # ---- multi-scale training ----
            size_rates = [0.75, 1, 1.25]
            loss_record2, loss_record3, loss_record4, loss_record5, loss_ewc_record6 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
            # for t, (train_loader, val_loader, test_loader) in enumerate(data_loaders):
            total_step = len(train_loader)
            for i, pack in enumerate(train_loader, start=1):
                # if i > 20 : # speed up (debug / quick run)
                #     break

                for rate in size_rates:
                    optimizer.zero_grad()
                    # ---- data prepare ----
                    images, gts = pack
                    images = Variable(images).cuda()
                    gts = Variable(gts).cuda()
                    # ---- rescale ----
                    trainsize = int(round(opt.trainsize*rate/32)*32)
                    if rate != 1:
                        images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    # ---- forward ----
                    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
                    # ---- loss function ----
                    loss5 = structure_loss(lateral_map_5, gts)
                    loss4 = structure_loss(lateral_map_4, gts)
                    loss3 = structure_loss(lateral_map_3, gts)
                    loss2 = structure_loss(lateral_map_2[t], gts)

                    if opt.approches == 'HNN':
                        if t > 0 :
                            if t == 1:
                                loss_ewc = cal_ewc_loss(model, model_old,fisher_information, opt.lamb_HNN)
                            else: 
                                loss_ewc = cal_ewc_loss(model, model_old,cognitive_map_HNN, opt.lamb_HNN)

                            loss =  loss2 +  loss3 +   loss4 +  loss5 +  loss_ewc   # TODO: try different weights for loss
                        else:
                            loss = loss2 + loss3 + loss4 + loss5
                    elif opt.approches == 'EWC':
                        if t > 0 :
                            loss_ewc = cal_ewc_loss(model, model_old,fisher_information, opt.lamb_HNN)
                            loss = loss2 + loss3 + loss4 + loss5 +  loss_ewc   # TODO: try different weights for loss
                        else:
                            loss = loss2 + loss3 + loss4 + loss5
                    elif opt.approches == 'BP':
                            loss = loss2 + loss3 + loss4 + loss5
                    else:
                        raise ValueError("未指定训练方法")

                    # ---- backward ----
                    loss.backward()
                    clip_gradient(optimizer, opt.clip)
                    optimizer.step()
                    # ---- recording loss ----
                    if rate == 1:
                        loss_record2.update(loss2.data, opt.batchsize)
                        loss_record3.update(loss3.data, opt.batchsize)
                        loss_record4.update(loss4.data, opt.batchsize)
                        loss_record5.update(loss5.data, opt.batchsize)

                        if opt.approches == 'EWC' or opt.approches == 'HNN':
                            if t > 0 :
                                loss_ewc_record6.update(loss_ewc.data, opt.batchsize)

                    

                # ---- train visualization ----
                if i % 5 == 0 or i == total_step:
                    print('Task [{}] {} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                        '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}, , lateral-6: {:0.4f}]'.
                        format(t ,datetime.now(), i_epoch, opt.epoch, i, total_step,
                                loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show(), loss_ewc_record6.show()))
                    
            # ++++ 新增：收集并保存损失数据 ++++
            # 获取各损失平均值
            avg_loss2 = loss_record2.show()
            avg_loss3 = loss_record3.show()
            avg_loss4 = loss_record4.show()
            avg_loss5 = loss_record5.show()
            avg_loss_ewc = loss_ewc_record6.show() if (opt.approches in ['EWC', 'HNN'] and t > 0) else 0.0

            # 写入CSV文件
            with open(loss_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                # 写入表头（如果文件为空）
                if os.stat(loss_file_path).st_size == 0:
                    writer.writerow(['task', 'epoch', 'loss2', 'loss3', 'loss4', 'loss5', 'loss_ewc'])
                # 写入当前任务和周期的损失数据
                writer.writerow([t, i_epoch, avg_loss2, avg_loss3, avg_loss4, avg_loss5, avg_loss_ewc])
                    
            if (i_epoch+1) %  epoch == 0:
                file_name = 'PraNet-{}-{}.pth'.format(t, i_epoch)
                torch.save(model.state_dict(), save_path + file_name)
                print('[Saving Snapshot:]', save_path + file_name)

        fisher_information = compute_fisher_information(model, train_loader,optimizer,t)
        fisher_dict[t] = fisher_information

        # save fisher_dict after training completes
        # fisher_save_path = os.path.join(save_path, 'fisher_dict.pth')
        # torch.save(fisher_dict, fisher_save_path)
        # print(f"fisher_dict saved to {fisher_save_path}")

        if t>0:
            # Aggregate past Fisher information into `cognitive_map_HNN` with decay.
            #
            # This block constructs a decayed aggregation of previously
            # computed Fisher importance estimates (stored in `fisher_dict`) to
            # form `cognitive_map_HNN`. The decay models the hippocampal-style memory
            # fading: older tasks contribute less importance. Critically,
            # this implements the paper's hippocampal learning paradigm where
            # the system captures both the order (task age) and the
            # importance (Fisher values) of previously learned tasks so that
            # EWC-style regularization can protect parameters according to
            # their episodic importance and recency.
            #
            # Implementation details:
            # - Start from zeros and accumulate decayed Fisher tensors.
            # - `decay = int(t - key)` represents the age (how many tasks
            #   ago the stored Fisher was computed); larger age -> stronger
            #   decay via `decay_tensor`.
            # - `decay_tensor` supports different decay modes; here 'linear'
            #   is used to attenuate past importance.
            cognitive_map_HNN = {name: torch.zeros_like(param) for name, param in fisher_information.items() }
            for key in list(fisher_dict.keys()):
                cognitive_map_HNN_mid = fisher_dict[key]
                # For each stored parameter tensor from past tasks, decay
                # its importance according to how old the task is, then add
                # to the running `cognitive_map_HNN` aggregate.
                for key2 in cognitive_map_HNN_mid.keys():
                    if cognitive_map_HNN_mid[key2] is not None:
                        # age of the saved Fisher (how many tasks since it was computed)
                        time = int(t - key)
                        # apply decay to simulate hippocampal fading of older
                        # episodic importance while preserving relative
                        # parameter importance within that episode
                        cognitive_map_HNN_mid[key2] = decay_tensor(time=time, tensor=cognitive_map_HNN_mid[key2], method='linear')
                        # accumulate into the aggregated HNN Fisher map
                        cognitive_map_HNN[key2] += cognitive_map_HNN_mid[key2]
        # Update old
        model_old = deepcopy(model)
        model_old.train()
        freeze_model(model_old) # Freeze the weights

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=40, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=256, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='PraNet_Res2Net')
    parser.add_argument('--task_num', type=int,
                        default=6, help='task number')
    parser.add_argument('--lamb_HNN', type=float,
                        default=1e5, help='loss_HNN Parameter coefficient')
    parser.add_argument('--approches', type=str,
                        default='EWC', help='choose the training method: EWC, HNN, BP')
    opt = parser.parse_args()



    for cycle in range(1):
        image_root = r'D:\PolypGen2021_MultiCenterData_v3'

        data_loaders = get_loaders(image_root, batchsize=opt.batchsize, trainsize=opt.trainsize,ratio_set=[0.8,0.2,0.])
        print(data_loaders)

        # ---- build models ----
        # torch.cuda.set_device(0)  # set your gpu device
        task_num = len(data_loaders)
        model = PraNet(task_num= task_num).cuda()
        
        params = model.parameters()
        optimizer = torch.optim.Adam(params, opt.lr)    

        print("#"*20, "Start Training", "#"*20)

        train(data_loaders, model, optimizer, opt.epoch)

