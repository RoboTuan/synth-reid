import torch
import torch.nn as nn


class SimLoss(nn.Module):
    """
    """

    def __init__(self, loss_type='feat_match'):
        super(SimLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.loss_type = loss_type

    def forward(self, feat_1, feat_2):
        if self.loss_type == 'feat_match':
            loss = self.mse(feat_1, feat_2)
        elif self.loss_type == 'gram_match':
            feat_1 = feat_1.flatten(2)
            feat_2 = feat_2.flatten(2)
            C, HW = feat_1.shape[1], feat_1.shape[2]
            G1 = torch.bmm(feat_1, torch.transpose(feat_1, 1, 2))
            G2 = torch.bmm(feat_2, torch.transpose(feat_2, 1, 2))
            loss = self.mse(G1, G2) / (C * HW)
        else:
            print(f"{self.loss_type} not implemented")
            raise NotImplementedError
        return loss
