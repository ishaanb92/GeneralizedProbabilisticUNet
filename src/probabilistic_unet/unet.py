from .unet_blocks import *
import torch.nn.functional as F

class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, initializers, apply_last_layer=True, padding=True, mc_dropout=False, dropout_rate=0.5):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            # Bottom-most encoding layer is #filters-2, #filters-1 is the bottleneck!
            if mc_dropout is True and i == len(self.num_filters) - 2:
                self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool, dropout=True, dropout_rate=dropout_rate))
            else:
                self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool, dropout=False))


        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            if mc_dropout is True and i == n:
                self.upsampling_path.append(UpConvBlock(input, output, initializers, padding, dropout=True, dropout_rate=dropout_rate))
            else:
                self.upsampling_path.append(UpConvBlock(input, output, initializers, padding, dropout=False))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)
            #nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            #nn.init.normal_(self.last_layer.bias)


    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])

        del blocks

        #Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)

        if self.apply_last_layer:
            x =  self.last_layer(x)

        return x

    def get_l2_params(self):
        # Get conv weights for encoder/decoder where dropout is applied
        encoder_weights = self.contracting_path[len(self.num_filters)-1].get_conv_params()
        decoder_weights = self.upsampling_path[0].get_conv_params()

        return encoder_weights+decoder_weights


