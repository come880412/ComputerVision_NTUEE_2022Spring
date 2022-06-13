import torch.functional as F
import torch.nn as nn

from torch.nn.modules.loss import CrossEntropyLoss

from torch.autograd import Variable
import torch
import warnings
warnings.filterwarnings("ignore")

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=[0.25, 0.75], softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.log_softmax = nn.LogSoftmax()

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = self.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class DiceFocalLoss(nn.Module):
    def __init__(self, args):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = args.dice_weight
        self.dice_loss = DiceLoss(n_classes = args.num_classes)
        self.focal_loss = FocalLoss()
    
    def forward(self, pred, target):
        loss_dice = self.dice_loss(pred, target, softmax = True)
        loss_focal = self.focal_loss(pred, target)
        loss = (1 - self.dice_weight) * loss_focal + self.dice_weight * loss_dice

        return loss

class DiceCeLoss(nn.Module):
    def __init__(self, args):
        super(DiceCeLoss, self).__init__()
        self.dice_weight = args.dice_weight
        self.dice_loss = DiceLoss(n_classes =  args.num_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        loss_dice = self.dice_loss(pred, target, softmax = True)
        loss_ce = self.ce_loss(pred, target)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, args):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = args.dice_weight
        self.dice_loss = DiceLoss(n_classes =  args.num_classes)
        self.ce_loss = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        loss_dice = self.dice_loss(pred, target, sigmoid = True)
        loss_ce = self.ce_loss(pred[:, 0, :, :], target.float())
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        return loss