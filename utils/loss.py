import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义了一个用于2D图像的交叉熵损失函数
# 这是一个用于语义分割人物的损失函数
class CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''

    def __init__(self, weight=None, ignore_label=255):
        # weight：一个1D权重向量，用于处理类不平衡问题。
        # ignore_label：用于忽略标签的值。
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. 
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        '''
        # 初始化当前的实例
        super().__init__()

        # self.loss = nn.NLLLoss2d(weight, ignore_index=255)
        self.loss = nn.NLLLoss(weight, ignore_index=ignore_label)
    # 输入outputs通过F.log_softmax函数传递到log-softmax层中，将输出转换为对数概率。
    # 接着，将得到的对数概率和目标targets一起传递给self.loss函数，该函数计算预测输出和目标输出之间的负对数似然损失。
    # 最后，返回损失值。
    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)

# 该代码实现了一个2维图像的Focal loss，是一种常用于解决类别不平衡问题的损失函数。
# Focal Loss通过调整难易样本的权重，对于容易分类的样本，降低其权重，让网络更加关注困难的样本。
# 其中，alpha是难易样本权重的平衡因子，gamma是难易样本权重的调整因子，weight是权重矩阵，
# ignore_index是忽略标签的类别编号。Focal Loss的实现是基于交叉熵损失CrossEntropyLoss的，
# 首先计算交叉熵损失的负对数概率，然后根据Focal Loss的公式计算损失值。
class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

# ProbOhemCrossEntropy2d是在语义分割任务中使用的一种损失函数，它使用了Online Hard Example Mining（OHEM）的技术来优化损失函数。
# OHEM通过选择困难样本进行训练，从而增强了模型的泛化能力。
# ignore_label：忽略的标签。表示在损失计算过程中要忽略的类别标签。
# reduction：损失的缩减方式。默认为'mean'，可以是'mean'、'sum'或'none'。
# thresh：用于选择难例的置信度阈值。
# min_kept：表示在损失计算过程中保留的最小像素数。
# down_ratio：表示输入图像的下采样率。
# use_weight：是否使用类别权重。
class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        # 在ProbOhemCrossEntropy2d类的构造函数中，使用super()函数调用父类(nn.Module)的构造函数。
        # 然后，使用构造函数的输入参数初始化了几个实例变量。
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            # logger.info('Labels: {}'.format(num_valid))
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~ valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~ valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)
