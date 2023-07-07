import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from utils.colorize_mask import cityscapes_colorize_mask, camvid_colorize_mask

# 实现了一个函数__init_weight，该函数用于对给定的神经网络中的卷积层和批归一化层进行初始化
def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # 该函数用于对神经网络模型的权重进行初始化。
    # module_list：神经网络模型的模块列表，其中包含需要初始化的所有层。
    # conv_init：初始化卷积层的方法。
    # norm_layer：规范化层的类型。
    # bn_eps：规范化层的 epsilon 参数。
    # bn_momentum：规范化层的 momentum 参数。
    # **kwargs：其他可选参数。
def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    # 判断module_list是否为list类型
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 该函数用于将模型对给定图像的预测输出与其地面真实值保存在指定目录中。
# output：模型的预测输出，可以是灰度图像或彩色图像
# gt：与输出相同输入图像的地面真实值
# img_name：输入图像的名称
# dataset：图像来自的数据集名称
# save_path：输出图像和地面真实值的保存目录路径
# output_grey：一个布尔标志，指示是否将输出图像保存为灰度格式
# output_color：一个布尔标志，指示是否将输出图像保存为彩色格式
# gt_color：一个布尔标志，指示是否将地面真实值图像保存为彩色格式
# 如果output_grey为True，则输出图像将保存为与输入图像同名的灰度PNG图像，保存在save_path指定的目录中。

def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=True, gt_color=False):
    if output_grey:
        output_grey = Image.fromarray(output)
        output_grey.save(os.path.join(save_path, img_name +'.png'))

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'camvid':
            output_color = camvid_colorize_mask(output)

        output_color.save(os.path.join(save_path, img_name + '_color.png'))

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'camvid':
            gt_color = camvid_colorize_mask(gt)

        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))

# 实现了计算神经网络模型的总参数量的功能
# 输入参数为一个神经网络模型，函数会遍历这个模型中的所有参数，对于每个参数，计算它的维度并相乘，
# 最终把所有参数的维度乘积累加起来，得到模型的总参数量。函数返回值为一个整数，表示模型的总参数量。
def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters
