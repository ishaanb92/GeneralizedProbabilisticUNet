"""
Implement a MixtureofGaussians (MoG) distribution that conforms to the torch.distributions API

The MixtureSameFamily implemented in Pytorch is not backprop friendly
    1. Does not have rsample() method
    2. Does not support relaxed versions of the Categorical distributions

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import torch
from torch.distributions import Distribution
from torch.distributions import MixtureSameFamily
from torch.distributions import RelaxedOneHotCategorical
from torch.distributions import constraints

class MixtureOfGaussians(MixtureSameFamily, Distribution):

    has_rsample = True

    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):

        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution

        assert(self._component_distribution.has_rsample is True)

        if not isinstance(self._mixture_distribution, RelaxedOneHotCategorical):
            raise ValueError("The Mixture distribution needs to be an instance of torch.distributions.Distribution.RelaxedOneHotCategorical")

        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._component_distribution.batch_shape[:-1]

        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                                  "compatible with `component_distribution'."
                                "batch_shape`({1})".format(mdbs, cdbs))

        # Check that the number of mixture component matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution component` ({0}) does not"
                             " equal `component_distribution.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self._num_component = km
        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)

        Distribution.__init__(self,
                              batch_shape=cdbs,
                              event_shape=event_shape,
                              validate_args=validate_args)

    def get_mixture_distribution(self):
        return self.mixture_distribution

    def rsample(self, sample_shape=torch.Size(), show_indices=False):

        # Shape : [B, n_components]
        mix_sample = self.mixture_distribution.rsample(sample_shape)


        # Straight-Through Gumble-Softmax: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
        # See also: https://wiki.lzhbrian.me/notes/differientiable-sampling-and-argmax
        # See: E. Jang, S. Gu, and B. Poole. Categorical Reparameterization with Gumbel-Softmax (2017), ICLR 2017
        # In the forward-pass, we get the one-hot vector (the actual sample cancels out) and in the backward pass
        # argmax does not have a contribute to the gradient but mix_sample does!
        # TODO: Verify gradient flow!!
        index = mix_sample.max(dim=-1, keepdim=True)[1]
        sample_mask = torch.zeros_like(mix_sample).scatter_(-1, index, 1.0)

        # Shape: [B, n_components]
        sample_mask = sample_mask - mix_sample.detach() + mix_sample

        # Shape: [B, n_components, z_dim]
        comp_samples = self.component_distribution.rsample(sample_shape)

        # Add a "fake" axis to the sample mask to broadcast the multiplication
        samples = torch.mul(comp_samples, sample_mask.unsqueeze(dim=-1))

        # The one-hot sample mask will zero-out all the rows
        # apart from the "selected" mixture component
        # So by summing along the columns, we recover the
        # sample from the "winning" Gaussian
        samples = torch.sum(samples, dim=-2)

        if show_indices is False:
            return samples
        else:
            return torch.cat([samples, index], dim=-1)







