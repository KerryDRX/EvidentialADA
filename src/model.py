import torch
from torch import nn
from config import cfg
from torchvision import models
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cfg.DATASET.NUM_CLASSES),
        )
        self.activation = self.activation_function()

    def forward(self, x):
        evidence = self.model(x)
        alpha = self.activation(evidence)
        return alpha

    def parameters_list(self):
        return [
            {'params': module.parameters(), 'lr_mult': 1. if name == 'fc' else 0.1}
            for name, module in self.model.named_children()
        ]
    
    def activation_function(self):
        if cfg.LOSS.ACTIVATION == 'relu':
            return lambda logits: F.relu(logits) + 1e-6
        elif cfg.LOSS.ACTIVATION == 'exp':
            return lambda logits: torch.exp(torch.clamp(logits, min=-10, max=10))
        elif cfg.LOSS.ACTIVATION == 'softplus':
            return lambda logits: F.softplus(logits)
        else:
            raise NotImplementedError(f'Activation not implemented: {cfg.LOSS.ACTIVATION}')
    