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

    def compute_kl_divergence(self, posterior_dist, prior_dist):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """

        try:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(posterior_dist,
                                      prior_dist)

        except NotImplementedError:
            # If the analytic KL divergence does not exists, use MC-approximation
            # See: 'APPROXIMATING THE KULLBACK LEIBLER DIVERGENCE BETWEEN GAUSSIAN MIXTURE MODELS' by Hershey and Olsen (2007)

            monte_carlo_terms = torch.zeros(size=(mc_samples, posterior_dist.batch_shape[0]),
                                                  dtype=torch.float32,
                                                  device=posterior_dist.rsample().device)

            # MC approximation of KL(q(z|x, y) || p(z|x)) = 1/N(\Sigma log(q(z) - log(p(z)))), z ~ q(z|x, y)
            for mc_iter in range(self.mc_samples):
                posterior_sample = posterior_dist.rsample()
                log_posterior_prob = posterior_dist.log_prob(posterior_sample)
                log_prior_prob = prior_dist.log_prob(posterior_sample)
                monte_carlo_terms[mc_iter, :] = log_posterior_prob - log_prior_prob

            # MC-approximation
            kl_div = torch.mean(monte_carlo_terms, dim=0)

        return kl_div

    def forward(self, input=None, target=None, prior_dist=None, posterior_dist=None):

        if prior_dist is not None:
            kl_loss = torch.mean(self.compute_kl_divergence(posterior_dist, prior_dist))
        else:
            kl_loss = 0.0

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
        loss['loss'] = recon_loss + self.beta*kl_loss
        loss['kl'] = kl_loss
        loss['reconstruction'] = recon_loss

        return loss
