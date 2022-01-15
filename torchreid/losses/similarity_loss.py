import torch
import torch.nn as nn


class SimLoss(nn.Module):
    """
    """

    def __init__(self):
        super(SimLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, feat_1, feat_2):
        # print(feat_1.shape, feat_2.shape)
        feat_1 = feat_1.flatten(2)
        feat_2 = feat_2.flatten(2)
        # print(feat_1.shape, feat_2.shape)
        C, HW = feat_1.shape[1], feat_1.shape[2]
        # print(C, HW)
        G1 = torch.bmm(feat_1, torch.transpose(feat_1, 1, 2))
        G2 = torch.bmm(feat_2, torch.transpose(feat_2, 1, 2))
        # print(G1.shape, G2.shape)
        loss = self.mse(G1, G2) / (C * HW)
        # print(loss.item())
        # print()
        return loss
