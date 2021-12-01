"""
PyTorch definition for the Probabilistic U-Net loss (ELBO loss with a KL-weight)

Author: Ishaan Bhat
Email: ishaan@isi.uu.nl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl

class ELBOLoss(nn.Module):

    def __init__(self, mode='ce', mc_samples=100, beta=10.0, class_weight=None):

        super(ELBOLoss, self).__init__()

        self.mc_samples = mc_samples
        self.mode = mode

        if mode == 'ce':
            self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        elif mode == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()

        self.beta = beta


    def forward(self, input=None, target=None, kld=0.0):

        n_tasks = target.shape[1]

        if self.mode == 'ce':
            target = torch.squeeze(target, dim=1)
            recon_loss = self.criterion(input=input,
                                        target=torch.argmax(target, dim=1))
        else:
            recon_loss = 0.0
            for task_id in range(n_tasks):
                recon_loss += self.criterion(input=input[:, task_id, ...],
                                             target=target[:, task_id, ...])

            recon_loss = recon_loss/n_tasks

        loss = {}
        loss['loss'] = recon_loss + self.beta*kld
        loss['kl'] = kld
        loss['reconstruction'] = recon_loss

        return loss
