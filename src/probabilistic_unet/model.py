#This code is based on: https://github.com/SimonKohl/probabilistic_unet

from .unet_blocks import *
from .unet import Unet
from .utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl, LowRankMultivariateNormal, RelaxedOneHotCategorical
from .mog import MixtureOfGaussians


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False,norm=True, label_channels=1):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += label_channels

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))
            if i < len(self.num_filters)-1 and norm == True:
                layers.append(nn.BatchNorm2d(output_dim))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, label_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False, n_components=1, temperature=0.1, norm=True):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.n_components = n_components
        self.temperature = temperature

        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'


        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior, norm=norm, label_channels=label_channels)

        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.n_components*self.latent_dim, (1,1), stride=1)
        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

        # 1x1 convolution to compute logits to parametrize the mixture distribution
        if self.n_components > 1:
            self.mixture_weights_conv = nn.Conv2d(num_filters[-1], self.n_components, (1, 1), stride=1)

        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0


    def forward(self, input, segm=None, one_hot=True):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm

            input = torch.cat((input, segm), dim=1)

            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu_log_sigma = mu_log_sigma.view(mu_log_sigma.shape[0], self.n_components, 2*self.latent_dim)

        if self.n_components == 1:
            dist = Independent(Normal(loc=mu_log_sigma[:, :, :self.latent_dim].squeeze(dim=1),
                                      scale=torch.exp(mu_log_sigma[:, :, self.latent_dim:].squeeze(dim=1))),1)
        else:
            logits = self.mixture_weights_conv(encoding) # Shape : [batch_size, n_components, 1, 1]
            logits = torch.squeeze(logits, dim=-1)
            logits = torch.squeeze(logits, dim=-1)

            cat_distribution = RelaxedOneHotCategorical(logits=logits,
                                                        temperature=torch.Tensor([self.temperature]).to(logits.device))

            comp_distribution = Independent(Normal(loc=mu_log_sigma[:, :, :self.latent_dim], scale=torch.exp(mu_log_sigma[:, :, self.latent_dim:])), 1)

            # Create a GMM
            dist = MixtureOfGaussians(mixture_distribution=cat_distribution,
                                      component_distribution=comp_distribution)

        assert(dist.batch_shape[0] == input.shape[0])

        return dist

class LowRankCovConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with low-rank approximation of covariance matrix.
    """

    def __init__(self, input_channels, label_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False, rank=1, n_components=1, temperature=0.1, norm=True):
        super(LowRankCovConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.rank = rank
        self.n_components = n_components
        self.temperature = temperature

        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'

        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior, norm=norm, label_channels=label_channels)


        # Low-rank approximation via covariance factors
        self.mean_log_sigma_op = nn.Conv2d(num_filters[-1], self.n_components*2*self.latent_dim, (1, 1), stride=1)
        self.cov_factor_op = nn.Conv2d(num_filters[-1], self.n_components*self.latent_dim*self.rank, (1, 1), stride=1)

        nn.init.kaiming_normal_(self.mean_log_sigma_op.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.mean_log_sigma_op.bias)

        nn.init.kaiming_normal_(self.cov_factor_op.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.cov_factor_op.bias)

        if self.n_components > 1:
            self.mixture_weights_conv = nn.Conv2d(num_filters[-1], self.n_components, (1, 1), stride=1)

        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

    def forward(self, input, segm=None, one_hot=True):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm

            input = torch.cat((input, segm), dim=1)

            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        # Mean over spatial dims
        encoding = torch.mean(encoding,
                              dim=(2, 3),
                              keepdim=True)

        mu_log_sigma = self.mean_log_sigma_op(encoding)
        mu_log_sigma = mu_log_sigma.squeeze(dim=-1).squeeze(dim=-1)
        mu_log_sigma = mu_log_sigma.view(mu_log_sigma.shape[0], self.n_components, 2*self.latent_dim)

        cov_factor = self.cov_factor_op(encoding)
        cov_factor = cov_factor.squeeze(dim=-1).squeeze(dim=-1)
        cov_factor = cov_factor.view(cov_factor.shape[0], self.n_components, self.latent_dim, self.rank)

        if self.n_components == 1:
            cov_factor = cov_factor.squeeze(dim=1)
            mu_log_sigma = mu_log_sigma.squeeze(dim=1)

            dist = LowRankMultivariateNormal(loc=mu_log_sigma[:, :self.latent_dim],
                                             cov_factor=cov_factor,
                                             cov_diag=torch.exp(mu_log_sigma[:, self.latent_dim:]))
        else:
            logits = self.mixture_weights_conv(encoding) # Shape : [batch_size, n_components, 1, 1]
            logits = torch.squeeze(logits, dim=-1)
            logits = torch.squeeze(logits, dim=-1)

            cat_distribution = RelaxedOneHotCategorical(logits=logits,
                                                        temperature=torch.Tensor([self.temperature]).to(logits.device))

            comp_distribution = LowRankMultivariateNormal(loc=mu_log_sigma[:, :, :self.latent_dim],
                                                          cov_factor=cov_factor,
                                                          cov_diag=torch.exp(mu_log_sigma[:, :, self.latent_dim:]))
            # Create a GMM
            dist = MixtureOfGaussians(mixture_distribution=cat_distribution,
                                      component_distribution=comp_distribution)


        assert(dist.batch_shape[0] == input.shape[0])

        return dist

class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes,
            no_convs_fcomb, initializers, use_tile=True,norm=False):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'
        if not use_tile:
            self.latent_broadcast = nn.Sequential(
                GatedConvTranspose2d(self.latent_dim, 64, 32, 1, 0),
                GatedConvTranspose2d(64, 64, 5, 1, 2),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, self.latent_dim, 5, 1, 2)
            )

        layers = []

        #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
        layers.append(nn.Conv2d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(no_convs_fcomb-2):
            if norm:
                layers.append(nn.BatchNorm2d(self.num_filters[0]))
            layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)


        self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

        if initializers['w'] == 'orthogonal':
            self.layers.apply(init_weights_orthogonal_normal)
            self.last_layer.apply(init_weights_orthogonal_normal)
        else:
            self.layers.apply(init_weights)
            self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])
        else:
            z = z.unsqueeze(2).unsqueeze(2)
            z = self.latent_broadcast(z)
        #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
        feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
        output = self.layers(feature_map)
        return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, label_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=4, beta=1.0, mc_dropout=False, dropout_rate=0.0, low_rank=False, rank=-1, n_components=1, temperature=0.1, norm=True, flow=False):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta
        self.z_prior_sample = 0
        self.mc_dropout = mc_dropout
        self.low_rank = low_rank
        self.n_components = n_components
        self.temperature = temperature
        self.flow = flow
        self.norm = norm
        if self.low_rank is True:
            if rank < 0: # Not initialized
                raise ValueError('Low-rank set to True but rank not specified')
            else:
                self.rank = rank


        # Main U-Net
        self.unet = Unet(self.input_channels,
                         self.num_classes,
                         self.num_filters,
                         self.initializers,
                         apply_last_layer=False,
                         padding=True,
                         norm=norm)

        # Prior Net
        if self.low_rank is False:
            self.prior = AxisAlignedConvGaussian(input_channels=self.input_channels,
                                                 label_channels=label_channels,
                                                 num_filters=self.num_filters,
                                                 no_convs_per_block=self.no_convs_per_block,
                                                 latent_dim=self.latent_dim,
                                                 initializers=self.initializers,
                                                 posterior=False,
                                                 n_components=self.n_components,
                                                 temperature=self.temperature,
                                                 norm=norm)
        else:
            self.prior = LowRankCovConvGaussian(input_channels=self.input_channels,
                                                label_channels=label_channels,
                                                num_filters=self.num_filters,
                                                no_convs_per_block=self.no_convs_per_block,
                                                latent_dim=self.latent_dim,
                                                initializers=self.initializers,
                                                rank=self.rank,
                                                posterior=False,
                                                n_components=self.n_components,
                                                temperature=self.temperature,
                                                norm=norm)

        # Posterior Net
        if self.low_rank is False:
            self.posterior = AxisAlignedConvGaussian(input_channels=self.input_channels,
                                                     label_channels=label_channels,
                                                     num_filters=self.num_filters,
                                                     no_convs_per_block=self.no_convs_per_block,
                                                     latent_dim=self.latent_dim,
                                                     initializers=self.initializers,
                                                     posterior=True,
                                                     n_components=self.n_components,
                                                     temperature=self.temperature,
                                                     norm=norm)
        else:
            self.posterior = LowRankCovConvGaussian(input_channels=self.input_channels,
                                                    label_channels=label_channels,
                                                    num_filters=self.num_filters,
                                                    no_convs_per_block=self.no_convs_per_block,
                                                    latent_dim=self.latent_dim,
                                                    initializers=self.initializers,
                                                    rank=self.rank,
                                                    posterior=True,
                                                    n_components=self.n_components,
                                                    temperature=self.temperature,
                                                    norm=norm)

        # 1x1 convolutions to merge samples from the posterior into the decoder output
        self.fcomb = Fcomb(self.num_filters,
                          self.latent_dim,
                          self.input_channels,
                          self.num_classes,
                          self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'},
                          use_tile=True,
                          norm=norm)


    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            if self.flow:
                self.log_det_j, self.z0, self.z, self.posterior_latent_space = self.posterior.forward(patch, segm)
            else:
                self.posterior_latent_space = self.posterior.forward(patch,segm)
                self.z = self.posterior_latent_space.rsample()
                self.z0 = self.z.clone()
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch,False)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        log_pz = self.prior_latent_space.log_prob(z_prior)
        #log_qz = self.posterior_latent_space.log_prob(z_prior)
        return self.fcomb.forward(self.unet_features,z_prior), log_pz

    def get_latent_space_distribution(self, mode='prior'):

        if mode == 'prior':
            return self.prior_latent_space
        elif mode == 'posterior':
            return self.posterior_latent_space
        else:
            raise ValueError('{} is an invalid mode'.format(mode))


    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space).sum()

        else:
            log_posterior_prob = self.posterior_latent_space.log_prob(self.z)
            log_prior_prob = self.prior_latent_space.log_prob(self.z)
            kl_div = (log_posterior_prob - log_prior_prob).sum()

        if self.flow:
            kl_div = kl_div - self.log_det_j.sum()

        return kl_div

    def elbo(self, segm, mask=None,use_mask = True, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        batch_size = segm.shape[0]
        self.kl = (self.kl_divergence(analytic=analytic_kl, calculate_posterior=False))

        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean,
                calculate_posterior=False, z_posterior=self.z)
        if use_mask:
            self.reconstruction = self.reconstruction*mask

        criterion = nn.BCEWithLogitsLoss(reduction='none')
        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return self.reconstruction, self.reconstruction_loss/batch_size, self.kl/batch_size,\
                -(self.reconstruction_loss + self.beta * self.kl)/batch_size


#    def forward(self, patch, segm, training=True, one_hot=True, mc_samples=1000):
#        """
#        Construct prior latent space for patch and run patch through UNet,
#        in case training is True also construct posterior latent space
#        """
#        # Get the distribution(s)
#        if training:
#            self.posterior_latent_space = self.posterior.forward(patch, segm, one_hot=one_hot)
#
#        self.prior_latent_space = self.prior.forward(patch)
#
#        # Get the U-net features
#        self.unet_features = self.unet.forward(patch,False)
#
#        if training:
#            # Create a label prediction by merging the unet_features and a sample from the posterior
#            reconstruction = self.sample(mode='posterior')
#        else:
#            # Create a label prediction by merging the unet_features and a sample from the posterior (at test-time)
#            reconstruction = self.sample(mode='prior')
#
#        # Save the distribution so that we can use it to compute the KL-term in the ELBO
#        if training:
#            kld = self.compute_kl_divergence(posterior_dist=self.posterior_latent_space,
#                                             prior_dist=self.prior_latent_space,
#                                             mc_samples=mc_samples)
#            kld = torch.mean(kld)
#        else:
#            kld = 0.0
#
#        return (reconstruction, kld)
#
#    # DEBUG function (Can be called only after a call to model.forward())
#    def get_latent_space_distribution(self, mode='prior'):
#        if mode == 'prior':
#            return self.prior_latent_space
#        elif mode == 'posterior':
#            return self.posterior_latent_space
#        else:
#            raise RuntimeError('Invalid mode {}'.format(mode))
#
#
#    @staticmethod
#    def compute_kl_divergence(posterior_dist, prior_dist, mc_samples=100):
#        """
#        Calculate the KL divergence between the posterior and prior KL(Q||P)
#        analytic: calculate KL analytically or via sampling from the posterior
#        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
#        """
#
#        try:
#            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
#            kl_div = kl.kl_divergence(posterior_dist,
#                                      prior_dist)
#
#        except NotImplementedError:
#            # If the analytic KL divergence does not exists, use MC-approximation
#            # See: 'APPROXIMATING THE KULLBACK LEIBLER DIVERGENCE BETWEEN GAUSSIAN MIXTURE MODELS' by Hershey and Olsen (2007)
#
#            # MC-approximation
#            posterior_samples = posterior_dist.rsample(sample_shape=torch.Size([mc_samples]))
#            log_posterior_prob = posterior_dist.log_prob(posterior_samples)
#            log_prior_prob = prior_dist.log_prob(posterior_samples)
#            monte_carlo_terms = log_posterior_prob - log_prior_prob
#            kl_div = torch.mean(monte_carlo_terms, dim=0)
#
#        return kl_div
#
#    # Sampling function that is used during training and testing
#    # Uses the u_net features computed in the forward() call
#    # Only the fcomb.forward() is run again
#    def sample(self, mode='prior'):
#        """
#        Sample from the prior latent space. Used in inference!
#
#        """
#        if mode == 'prior':
#            z_sample = self.prior_latent_space.rsample()
#        elif mode == 'posterior':
#            z_sample = self.posterior_latent_space.rsample()
#
#        return self.fcomb.forward(self.unet_features, z_sample)
