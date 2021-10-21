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
    def __init__(self, input_channels, label_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
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

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, label_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False, n_components=1):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.n_components = n_components

        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'

        self.conv_layer = nn.ModuleList()

        self.encoder = Encoder(self.input_channels, label_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)

        for mix_component in range(self.n_components):
            self.conv_layer.append(nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1))

        if self.n_components > 1:
            self.mixture_weights_conv = nn.Conv2d(num_filters[-1], self.n_components, (1, 1), stride=1)

        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        for conv_op in self.conv_layer:
            nn.init.kaiming_normal_(conv_op.weight, mode='fan_in', nonlinearity='relu')
            nn.init.normal_(conv_op.bias)

    def forward(self, input, segm=None, one_hot=True):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm

            if one_hot is True:
                input = torch.cat((input, segm[:, :, 1, ...]), dim=1)
            else:
                input = torch.cat((input, segm), dim=1)

            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        mu = torch.zeros(size=(encoding.shape[0], self.n_components, self.latent_dim),
                         dtype=encoding.dtype,
                         device=encoding.device)

        log_sigma = torch.zeros(size=(encoding.shape[0], self.n_components, self.latent_dim),
                                dtype=encoding.dtype,
                                device=encoding.device)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        for mix_component, conv_op in enumerate(self.conv_layer):

            mu_log_sigma = conv_op(encoding)
            #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

            mu[:, mix_component, :] = mu_log_sigma[:, :self.latent_dim]
            log_sigma[:, mix_component, :] = mu_log_sigma[:, self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        # Define the torch.distribution so that samples can be generated!
        if self.n_components == 1:
            dist = Independent(Normal(loc=mu.squeeze(dim=1),
                                      scale=torch.exp(log_sigma.squeeze(dim=1))),1)
        else:
            # TODO: Understand the temperature parameter and maybe include it in the hyper-parameter optimization!
            logits = self.mixture_weights_conv(encoding) # Shape : [batch_size, n_components, 1, 1]
            logits = torch.squeeze(logits, dim=-1)
            logits = torch.squeeze(logits, dim=-1)

            cat_distribution = RelaxedOneHotCategorical(logits=logits,
                                                        temperature=torch.Tensor([0.5]).to(logits.device))

            comp_distribution = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)

            # Create a GMM
            dist = MixtureOfGaussians(mixture_distribution=cat_distribution,
                                      component_distribution=comp_distribution)

        assert(dist.batch_shape[0] == input.shape[0])

        return dist

class LowRankCovConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with low-rank approximation of covariance matrix.
    """
    def __init__(self, input_channels, label_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False, rank=1, n_components=1):
        super(LowRankCovConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.rank = rank
        self.n_components = n_components

        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'

        self.encoder = Encoder(self.input_channels, label_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)



        self.mean_op = nn.ModuleList()
        self.log_cov_diag_op = nn.ModuleList()
        self.cov_factor_op = nn.ModuleList()

        for mix_component in range(n_components):
            self.mean_op.append(nn.Conv2d(num_filters[-1], self.latent_dim, (1, 1), stride=1))
            self.log_cov_diag_op.append(nn.Conv2d(num_filters[-1], self.latent_dim, (1, 1), stride=1))
            self.cov_factor_op.append(nn.Conv2d(num_filters[-1], self.latent_dim*self.rank, (1, 1), stride=1))

        if self.n_components > 1:
            self.mixture_weights_conv = nn.Conv2d(num_filters[-1], self.n_components, (1, 1), stride=1)

        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        # Initialize parameters

        for mean, log_cov_diag, cov_factor in zip(self.mean_op, self.log_cov_diag_op, self.cov_factor_op):
            nn.init.kaiming_normal_(mean.weight, mode='fan_in', nonlinearity='relu')
            nn.init.normal_(mean.bias)

            nn.init.kaiming_normal_(log_cov_diag.weight, mode='fan_in', nonlinearity='relu')
            nn.init.normal_(log_cov_diag.bias)

            nn.init.kaiming_normal_(cov_factor.weight, mode='fan_in', nonlinearity='relu')
            nn.init.normal_(cov_factor.bias)

    def forward(self, input, segm=None, one_hot=True):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm

            if one_hot is True:
                input = torch.cat((input, segm[:, :, 1, ...]), dim=1)
            else:
                input = torch.cat((input, segm), dim=1)

            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        # Mean over spatial dims
        encoding = torch.mean(encoding,
                              dim=(2, 3),
                              keepdim=True)

        # Squeeze operations to remove singleton dimensions

        mu_mixture = torch.zeros(size=(encoding.shape[0], self.n_components, self.latent_dim),
                                 dtype=encoding.dtype,
                                 device=encoding.device)

        cov_diag_mixture = torch.zeros(size=(encoding.shape[0], self.n_components, self.latent_dim),
                                       dtype=encoding.dtype,
                                       device=encoding.device)

        cov_factor_mixture = torch.zeros(size=(encoding.shape[0], self.n_components, self.latent_dim, self.rank),
                                         dtype=encoding.dtype,
                                         device=encoding.device)

        for mix_component, (mean, log_cov_diag, cov_factor) in enumerate(zip(self.mean_op, self.log_cov_diag_op, self.cov_factor_op)):
            mu = mean(encoding)
            mu = torch.squeeze(mu, dim=-1)
            mu = torch.squeeze(mu, dim=-1)
            mu_mixture[:, mix_component, ...] = mu

            cov_diag = log_cov_diag(encoding)
            cov_diag = torch.squeeze(cov_diag, dim=-1)
            cov_diag = torch.squeeze(cov_diag, dim=-1)
            cov_diag = torch.exp(cov_diag)
            cov_diag_mixture[:, mix_component, ...] = cov_diag

            cov_factor = cov_factor(encoding)
            cov_factor = torch.squeeze(cov_factor, dim=-1)
            cov_factor = torch.squeeze(cov_factor, dim=-1)
            # Change view to get the shape: [batch size, self.latent_dim, self.rank]
            cov_factor = cov_factor.view(cov_factor.shape[0], self.latent_dim, self.rank)
            cov_factor_mixture[:, mix_component, ...] = cov_factor


        if self.n_components == 1:
            dist = LowRankMultivariateNormal(loc=mu_mixture.squeeze(dim=1),
                                             cov_factor=cov_factor_mixture.squeeze(dim=1),
                                             cov_diag=cov_diag_mixture.squeeze(dim=1))
        else:
            # TODO: Understand the temperature parameter and maybe include it in the hyper-parameter optimization!
            logits = self.mixture_weights_conv(encoding) # Shape : [batch_size, n_components, 1, 1]
            logits = torch.squeeze(logits, dim=-1)
            logits = torch.squeeze(logits, dim=-1)

            cat_distribution = RelaxedOneHotCategorical(logits=logits,
                                                        temperature=torch.Tensor([0.5]).to(logits.device))

            comp_distribution = LowRankMultivariateNormal(loc=mu_mixture,
                                                          cov_factor=cov_factor_mixture,
                                                          cov_diag=cov_diag_mixture)
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
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
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

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
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

    def __init__(self, input_channels=1, label_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=4, beta=10.0, mc_dropout=False, dropout_rate=0.5, low_rank=False, rank=-1, n_components=1):
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

        if self.low_rank is True:
            if rank < 0: # Not initialized
                self.rank = self.latent_dim # Full covariance matrix
            else:
                self.rank = rank


        # Main U-Net
        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers, apply_last_layer=False, padding=True, mc_dropout=mc_dropout, dropout_rate=dropout_rate)

        # Prior Net
        if self.low_rank is False:
            self.prior = AxisAlignedConvGaussian(input_channels=self.input_channels,
                                                 label_channels=label_channels,
                                                 num_filters=self.num_filters,
                                                 no_convs_per_block=self.no_convs_per_block,
                                                 latent_dim=self.latent_dim,
                                                 initializers=self.initializers,
                                                 posterior=False,
                                                 n_components=self.n_components)
        else:
            self.prior = LowRankCovConvGaussian(input_channels=self.input_channels,
                                                label_channels=label_channels,
                                                num_filters=self.num_filters,
                                                no_convs_per_block=self.no_convs_per_block,
                                                latent_dim=self.latent_dim,
                                                initializers=self.initializers,
                                                rank=self.rank,
                                                posterior=False,
                                                n_components=self.n_components)

        # Posterior Net
        if self.low_rank is False:
            self.posterior = AxisAlignedConvGaussian(input_channels=self.input_channels,
                                                     label_channels=label_channels,
                                                     num_filters=self.num_filters,
                                                     no_convs_per_block=self.no_convs_per_block,
                                                     latent_dim=self.latent_dim,
                                                     initializers=self.initializers,
                                                     posterior=True,
                                                     n_components=self.n_components)
        else:
            self.posterior = LowRankCovConvGaussian(input_channels=self.input_channels,
                                                    label_channels=label_channels,
                                                    num_filters=self.num_filters,
                                                    no_convs_per_block=self.no_convs_per_block,
                                                    latent_dim=self.latent_dim,
                                                    initializers=self.initializers,
                                                    rank=self.rank,
                                                    posterior=True,
                                                    n_components=self.n_components)

        # 1x1 convolutions to merge samples from the posterior into the decoder output
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True)

    def forward(self, patch, segm, training=True, one_hot=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm, one_hot=one_hot)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch,False)
        self.l2_params = self.unet.get_l2_params()


    def sample(self):
        """
        Sample from the prior latent space. Used in inference!

        """
        z_prior = self.prior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_prior)



    def reconstruct(self, use_posterior_mean=False, z=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z = self.posterior_latent_space.loc
        elif z is None:
            z = self.posterior_latent_space.rsample()

        return self.fcomb.forward(self.unet_features, z)


    def kl_divergence(self, mc_samples=100):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """

        try:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space,
                                      self.prior_latent_space)
        except NotImplementedError:
            # If the analytic KL divergence does not exists, use MC-approximation
            # See: 'APPROXIMATING THE KULLBACK LEIBLER DIVERGENCE BETWEEN GAUSSIAN MIXTURE MODELS' by Hershey and Olsen (2007)
            monte_carlo_terms = torch.zeros(size=(mc_samples, self.posterior_latent_space.batch_shape[0]),
                                                  dtype=self.unet_features.dtype,
                                                  device=self.unet_features.device)
            for mc_iter in range(mc_samples):
                posterior_sample = self.posterior_latent_space.rsample() #FIXME: Implement rsample() method
                log_posterior_prob = self.posterior_latent_space.log_prob(posterior_sample)
                log_prior_prob = self.prior_latent_space.log_prob(posterior_sample)
                monte_carlo_terms[mc_iter, :] = log_posterior_prob - log_prior_prob

            # MC-approximation
            kl_div = torch.mean(monte_carlo_terms, dim=0)

        return kl_div

    def compute_loss(self, segm, train=True, class_weight=None):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        # Expected shape of ref labels (segm) : [B x n_annotations x n_tasks x 2 x h x w]
        n_tasks = segm.shape[2]
        n_annotations = segm.shape[1]

        if n_tasks == 1: # Squeeze the labels
            criterion = nn.CrossEntropyLoss(weight=class_weight) # Softmax + NLL
        else:
            criterion = nn.BCEWithLogitsLoss() # Sigmoid + BCE (per-task)

        kl_loss = 0
        reconstruction_loss = 0

        # Number of samples generated == number of annotations for the image (so we learn all the modes)
        for anno in range(n_annotations):

            # The posterior latent space is generated by the mean segmentation over annotations!
            # We assume samples from the posterior will learn to approximate the "modes" of the labels
            if train is True:
                z_posterior = self.posterior_latent_space.rsample()

                # KL-divergence between z_prior and z_posterior
                kl_loss += torch.mean(self.kl_divergence())

                # Output generated from the posterior
                reconstruction = self.reconstruct(use_posterior_mean=False,
                                                  z=z_posterior)

            else: # For validation loss, sample from z_prior i.e. the latent space conditioned only on the data
                z_prior = self.prior_latent_space.rsample()
                reconstruction = self.reconstruct(use_posterior_mean=False,
                                                  z=z_prior)

            for task in range(n_tasks):
                if n_tasks == 1:
                    reconstruction_loss += criterion(input=reconstruction,
                                                     target=torch.argmax(segm[:, anno, task, ...], dim=1))
                else:
                    reconstruction_loss += criterion(input=reconstruction[:, task, ...],
                                                     target=segm[:, anno, task, ...])


        reconstruction_loss = reconstruction_loss/(n_annotations*n_tasks)
        kl_loss = kl_loss/n_annotations


        loss_dict = {}
        loss_dict['loss'] = (reconstruction_loss + self.beta*kl_loss)

        loss_dict['reconstruction'] = reconstruction_loss
        loss_dict['kl'] = kl_loss


        return loss_dict

