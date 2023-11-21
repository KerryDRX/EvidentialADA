import torch
from config import cfg


def kld(alpha):
    # KL divergence
    ones = torch.ones((1, alpha.shape[1])).cuda()
    sum_alpha = alpha.sum(dim=1)
    loss_KL = (
        sum_alpha.lgamma()
        - ones.sum(dim=1).lgamma()
        - alpha.lgamma().sum(dim=1)
        + ((alpha - 1) * (alpha.digamma() - sum_alpha.unsqueeze(1).digamma())).sum(dim=1)
    ).sum() / alpha.shape[0]
    return loss_KL

def loss_nll(alpha, y):
    # EDL: negative log likelihood
    S = alpha.sum(dim=1, keepdim=True)
    return (y * (S.log() - alpha.log())).sum() / alpha.shape[0]

def loss_ce(alpha, y):
    # EDL: cross entropy
    S = alpha.sum(dim=1, keepdim=True)
    return (y * (S.digamma() - alpha.digamma())).sum() / alpha.shape[0]

def loss_sos(alpha, y):
    # EDL: sum of squares
    S = alpha.sum(dim=1, keepdim=True)
    err = ((y - (alpha / S)) ** 2).sum(dim=1, keepdim=True)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1, keepdim=True)
    return (err + var).sum() / alpha.shape[0]

class EDL_Loss(torch.nn.Module):
    def __init__(self, loss_fn, regularization):
        super(EDL_Loss, self).__init__()
        self.loss_fn = {
            'nll': loss_nll,
            'ce': loss_ce,
            'sos': loss_sos,
        }[loss_fn]
        self.regularization = regularization
    
    def forward(self, alpha, labels):
        y = torch.eye(alpha.shape[1]).cuda()[labels]
        loss = self.loss_fn(alpha, y)
        if self.regularization:
            loss += kld(y + (1.0 - y) * alpha) / cfg.DATASET.NUM_CLASSES
        return loss
