import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps    = 1e-9


    def forward(self, output_1, output_2, target, size_average=True):
        distances = (output_2 - output_1).pow(2).sum(1) # squared distances
        losses    = 0.5 * (target.float() * distances + 
                            (1 + -1*target).float() * F.relu(self.margin -(distances+self.eps).sqrt()).pow(2))
                        
        return losses.mean() if size_average else losses.sum()


    
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    
    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses            = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()