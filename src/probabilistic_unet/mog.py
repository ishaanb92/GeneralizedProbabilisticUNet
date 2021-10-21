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

    def rsample(self, sample_shape=torch.Size()):
        sample_len = len(sample_shape)
        batch_len = len(self.batch_shape)
        gather_dim = sample_len + batch_len
        es = self.event_shape

        mix_sample = self.mixture_distribution.rsample(sample_shape)

        # Straight-Through Gumble-Softmax -- Take the argmax in the forward pass and use the Gumble-Softmax during backpropagation
        # See: E. Jang, S. Gu, and B. Poole. Categorical Reparameterization with Gumbel-Softmax (2017), ICLR 2017
        mix_sample = torch.argmax(mix_sample, dim=-1)

        mix_shape = mix_sample.shape

        comp_samples = self.component_distribution.rsample(sample_shape)

        # Gather along the k dimension
        mix_sample_r = mix_sample.reshape(
            mix_shape + torch.Size([1]*(len(es) + 1)))

        mix_sample_r = mix_sample_r.repeat(
            torch.Size([1]*len(mix_shape)) + torch.Size([1]) + es)


        samples = torch.gather(comp_samples, gather_dim, mix_sample_r)

        return samples.squeeze(gather_dim)




