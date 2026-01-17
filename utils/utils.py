import torch
import numpy as np
# from thop import profile
# from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    Clip gradients to stabilize training.

    :param optimizer: optimizer instance
    :param grad_clip: threshold for clipping gradients
    :return: None
    """
    for group in optimizer.param_groups:  # iterate over optimizer parameter groups
        for param in group['params']:  # iterate over all parameters in the group
            if param.grad is not None:  # if the parameter has gradients
                param.grad.data.clamp_(-grad_clip, grad_clip)  # clamp gradients to [-grad_clip, grad_clip]


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    # AvgMeter tracks and computes the running average of losses.
    # This class is useful for recording and displaying loss trends during training.
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        # show method: compute and return the average of the most recent `num` loss values.
        if len(self.losses) == 0:
            return 0  # return 0 if no losses recorded (or another suitable default)
        recent_losses = self.losses[max(len(self.losses) - self.num, 0):]
        return torch.mean(torch.tensor(recent_losses))


# def CalParams(model, input_tensor):
#     """
#     Usage:
#         Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
#     Necessarity:
#         from thop import profile
#         from thop import clever_format
#     :param model:
#     :param input_tensor:
#     :return:
#     """
#     flops, params = profile(model, inputs=(input_tensor,))
#     flops, params = clever_format([flops, params], "%.3f")
#     print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return