from scipy.ndimage.morphology import distance_transform_edt as edt
import torch
import numpy as np
import torch.nn as nn

import medpy
import medpy.metric
def cal_hd_hd95(inputs_, targets, if95 =True):
    batch_size = inputs_.size(0)
    
    hds = []

    for i in range(batch_size):
        inputs = inputs_[i,0].clone().cpu().detach().numpy()
        target = targets[i,0].clone().cpu().detach().numpy()
     
        inputs[inputs>0.5] =1
        inputs[inputs<=0.5]=0

        if if95 ==True:
            hd =  medpy.metric.binary.hd95(result=inputs, reference = target)
        else:
            hd = medpy.metric.binary.hd(result=inputs, reference = target)

        hds.append(hd)

    return hds

class HausdorffDistance_incase:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.max(distances[indexes]))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()

        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
        ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
        ).float()

        return torch.max(right_hd, left_hd)

class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.max(distances[indexes]))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        batch_size = pred.shape[0]
        results = []

        for i in range(batch_size):
            single_pred = (pred[i] > 0.5).byte()
            single_target = (target[i] > 0.5).byte()

            right_hd = torch.from_numpy(
                self.hd_distance(single_pred.cpu().numpy(), single_target.cpu().numpy())
            ).float()

            left_hd = torch.from_numpy(
                self.hd_distance(single_target.cpu().numpy(), single_pred.cpu().numpy())
            ).float()

            results.append(torch.max(right_hd, left_hd))

        return torch.stack(results)
    

class HausdorffDistance95:
    def hd_distance(self, x: np.ndarray, y: np.ndarray, percentile: float = 95) -> np.ndarray:
        """Calculate the directed Hausdorff distance at a specified percentile."""
        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])
        
        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))
        all_distances = distances[indexes]
        
        return np.array([np.percentile(all_distances, percentile)])

    def compute(self, pred: torch.Tensor, target: torch.Tensor, percentile: float = 95) -> torch.Tensor:
        """Compute the symmetric Hausdorff distance at the 95th percentile for a batch of predictions."""
        assert pred.shape[1] == 1 and target.shape[1] == 1, "Only binary channel supported"

        batch_size = pred.shape[0]
        results = []

        for i in range(batch_size):
            single_pred = (pred[i] > 0.5).byte()
            single_target = (target[i] > 0.5).byte()

            right_hd = torch.from_numpy(
                self.hd_distance(single_pred.cpu().numpy(), single_target.cpu().numpy(), percentile)
            ).float()

            left_hd = torch.from_numpy(
                self.hd_distance(single_target.cpu().numpy(), single_pred.cpu().numpy(), percentile)
            ).float()

            results.append(torch.max(right_hd, left_hd))

        return torch.stack(results)
    
class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray, device: torch.device) -> torch.Tensor:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return torch.from_numpy(field).float().to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
        device = pred.device  # Ensures all tensors will be on the same device as the input
        assert pred.dim() in (4, 5), "Only 2D and 3D supported"
        assert pred.dim() == target.dim(), "Prediction and target need to be of same dimension"
        
        pred_dt = self.distance_field(pred.cpu().numpy(), device)
        target_dt = self.distance_field(target.cpu().numpy(), device)

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss

class Loss_all(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Loss_all, self).__init__()

    def forward(self, inputs_, targets, smooth=1):
       
        inputs = inputs_.clone()  
        # inputs = F.sigmoid(inputs)   
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        
        
        T_ = targets[inputs==targets]
        F_ = targets[inputs!=targets]
        
        TP = T_.sum()  #  TP
        TN = len(T_) - T_.sum()  #  TN
        
        FP = F_.sum() # FP
        FN = len(F_)-F_.sum() # FN
        
        precision= TP/(TP+FP)
        recall =  TP/(TP+FN)
        FPR = FP/(FP+TN)
        FNR = FN/(FN+TP)
  
        return precision,recall,FPR,FNR


class Loss_all_batch(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Loss_all_batch, self).__init__()

    def safe_div(self, num, denom):
        return num / denom if denom else 0.0

    def forward(self, inputs_, targets, smooth=1):
        batch_size = inputs_.size(0)
        
        # Initialize lists to store metrics for each batch
        precisions, recalls, FPRs, FNRs = [], [], [], []

        for i in range(batch_size):
            inputs = inputs_[i].clone()
            inputs = inputs.view(-1)
            targets_batch = targets[i].view(-1)
            inputs[inputs >= 0.5] = 1
            inputs[inputs < 0.5] = 0

            T_ = targets_batch[inputs == targets_batch]
            F_ = targets_batch[inputs != targets_batch]

            TP = T_.sum().float()  # TP
            TN = len(T_) - T_.sum().float()  # TN

            FP = F_.sum().float()  # FP
            FN = len(F_) - F_.sum().float()  # FN

            precision = self.safe_div(TP, (TP + FP))
            recall = self.safe_div(TP, (TP + FN))
            FPR = self.safe_div(FP, (FP + TN))
            FNR = self.safe_div(FN, (FN + TP))

            precisions.append(precision)
            recalls.append(recall)
            FPRs.append(FPR)
            FNRs.append(FNR)

        # Convert lists to tensors
        precisions = torch.tensor(precisions).mean()
        recalls = torch.tensor(recalls).mean()
        FPRs = torch.tensor(FPRs).mean()
        FNRs = torch.tensor(FNRs).mean()

        return precisions, recalls, FPRs, FNRs


class Loss_all_batch_each(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Loss_all_batch_each, self).__init__()

    def safe_div(self, num, denom):
        return num / denom if denom else 0.0

    def forward(self, inputs_, targets, smooth=1):
        batch_size = inputs_.size(0)
        
        # Initialize lists to store metrics for each batch
        precisions, recalls, FPRs, FNRs = [], [], [], []

        for i in range(batch_size):
            inputs = inputs_[i].clone()
            inputs = inputs.view(-1)
            targets_batch = targets[i].view(-1)
            inputs[inputs >= 0.5] = 1
            inputs[inputs < 0.5] = 0

            T_ = targets_batch[inputs == targets_batch]
            F_ = targets_batch[inputs != targets_batch]

            TP = T_.sum().float()  # TP
            TN = len(T_) - T_.sum().float()  # TN

            FP = F_.sum().float()  # FP
            FN = len(F_) - F_.sum().float()  # FN

            precision = self.safe_div(TP, (TP + FP))
            recall = self.safe_div(TP, (TP + FN))
            FPR = self.safe_div(FP, (FP + TN))
            FNR = self.safe_div(FN, (FN + TP))

            precisions.append(precision)
            recalls.append(recall)
            FPRs.append(FPR)
            FNRs.append(FNR)

        # Convert lists to tensors
        precisions = torch.tensor(precisions)
        recalls = torch.tensor(recalls)
        FPRs = torch.tensor(FPRs)
        FNRs = torch.tensor(FNRs)

        return precisions, recalls, FPRs, FNRs
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs_, targets, smooth=1,sig=False):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs_.clone()  
        if sig==True:
            inputs = torch.sigmoid(inputs) 

        # inputs[inputs>=0.5] = 1
        # inputs[inputs<0.5] = 0
        #flatten label and prediction tensors
    
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceLoss_batch(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_batch, self).__init__()

    def forward(self, inputs_, targets, smooth=1, sig=False):
        # Apply sigmoid if specified
        if sig:
            inputs_ = torch.sigmoid(inputs_)

        # Initialize dice score list
        dice_scores = []

        # Loop through the batch
        for i in range(inputs_.size(0)):
            inputs = inputs_[i].view(-1)
            targets_batch = targets[i].view(-1)
            
            intersection = (inputs * targets_batch).sum()
            dice = (2. * intersection + smooth) / (inputs.sum() + targets_batch.sum() + smooth)
            
            dice_scores.append(1 - dice)

        # Convert list to tensor and return mean dice loss
        dice_scores = torch.tensor(dice_scores)
        
        return dice_scores.mean()
    

class DiceLoss_batch_each(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_batch_each, self).__init__()

    def forward(self, inputs_, targets, smooth=1, sig=False):
        # Apply sigmoid if specified
        if sig:
            inputs_ = torch.sigmoid(inputs_)

        # Initialize dice score list
        dice_scores = []

        # Loop through the batch
        for i in range(inputs_.size(0)):
            inputs = inputs_[i].view(-1)
            targets_batch = targets[i].view(-1)
            
            intersection = (inputs * targets_batch).sum()
            dice = (2. * intersection + smooth) / (inputs.sum() + targets_batch.sum() + smooth)
            
            dice_scores.append(1 - dice)

        # Convert list to tensor and return mean dice loss
        dice_scores = torch.tensor(dice_scores)
        
        return dice_scores
    