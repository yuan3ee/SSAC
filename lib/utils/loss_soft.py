import torch
import logging
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from .online_label_smooth import OnlineLabelSmoothing

class CELoss(nn.Module):
    def __init__(self, label_smoothing=None, class_num=6):
        super(CELoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.class_num = class_num

    def forward(self, pred, target):
        eps = 1e-12

        if self.label_smoothing is not None:
            logprobs = F.log_softmax(pred, dim=1)
            target = F.one_hot(target.to(torch.int64), self.class_num)
            target = target.transpose(1, -1)
            target = torch.squeeze(target)

            # label smoothing
            target = torch.clamp(target.float(), min=self.label_smoothing/(self.class_num-1), max=1.0-self.label_smoothing)
            loss = -1 * torch.sum(target*logprobs, 1)

        else:
            loss = -1 * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))

        return loss.mean()

criterion = CELoss(label_smoothing=0.1, class_num=6)
# criterion = OnlineLabelSmoothing(alpha=0.5, n_classes=6, smoothing=0.1)

class antiCrossEntropyST(nn.Module):
    def __init__(self, config, D_out_z, ignore_label=-1, thres=0.9):
        super(antiCrossEntropyST, self).__init__()
        self.config = config
        self.ignore_label = ignore_label
        self.D_out_z = D_out_z
        self.thres = thres
        self.criterion = OnlineLabelSmoothing(alpha=0.5, n_classes=6, smoothing=0.1)


    def forward(self, predict, target, weight=None, epoch=False):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()

        predictProbability = F.softmax(predict, dim=1)
        antiPredictProbability = (1.0 - predictProbability)
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        antiPredictProbability = antiPredictProbability.transpose(1, 2).transpose(2, 3).contiguous()
        antiPredictProbability = antiPredictProbability[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        antiPredictProbabilityLog = torch.log(antiPredictProbability)
        # antiLoss = F.nll_loss(antiPredictProbabilityLog, target, reduction='none')
        antiLoss = self.criterion(antiPredictProbabilityLog, target)
        if epoch is True:
            self.criterion.next_epoch()
        if self.config.LOSS.DYNAMIC and self.D_out_z is not None:
            antiLossWeight = self.D_out_z[:, 0, :, :][target_mask]
            antiLoss = antiLossWeight * antiLoss
        return antiLoss.mean()


class CrossEntropyST(nn.Module):
    def __init__(self, config, D_out_z, ignore_label=-1):
        super(CrossEntropyST, self).__init__()
        self.config = config
        self.ignore_label = ignore_label
        self.D_out_z = D_out_z
        self.criterion = OnlineLabelSmoothing(alpha=0.5, n_classes=6, smoothing=0.1)


    def forward(self, predict, target, weight=None, epoch=False):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = self.criterion(predict, target)
        if epoch is True:
            self.criterion.next_epoch()

        # loss = F.cross_entropy(predict, target, weight=weight, reduction='none')
        if self.config.LOSS.DYNAMIC and self.D_out_z is not None:
            LossWeight = self.D_out_z[:, 0, :, :][target_mask]
            loss = LossWeight * loss
        return loss.mean()


class BCEWithLogitsLoss2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=-1):
        super(BCEWithLogitsLoss2d, self).__init__()
        # self.size_average = size_average
        self.ignore_label = ignore_label


    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, reduction='mean')
        return loss


class CrossEntropy(nn.Module):
    def __init__(self, config, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.config = config
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )


    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=self.config.MODEL.ALIGN_CORNERS)
        loss = self.criterion(score, target)
        return loss


    def forward(self, score, target):
        weights = self.config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)
        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


class OhemCrossEntropy(nn.Module):
    def __init__(self, config, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.config = config
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=self.config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=self.config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if self.config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = self.config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])
