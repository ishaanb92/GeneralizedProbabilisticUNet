### Probabilistic U-Net with MC-Dropout

The code in this repository was developed for the [QUBIQ 2021 challenge](https://qubiq21.grand-challenge.org/). We extend the [Probabilistic U-Net](https://proceedings.neurips.cc/paper/2018/hash/473447ac58e1cd7e96172575f48dca3b-Abstract.html) to estimate model uncertainty (in addition to data uncertainty) using the popular [MC-Dropout](http://proceedings.mlr.press/v48/gal16.html) technique. This approach is inspired from [Hu et al., 2019](http://arxiv.org/abs/1907.01949) that uses
[variational dropout](https://papers.nips.cc/paper/2015/file/bc7316929fe1545bf0b98d114ee3ecb8-Paper.pdf) within the Probabilistic U-Net framework to estimate the model uncertainty.

The code for the Probabilistic U-Net has been forked from [here](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch). We have restructured the code to enable import as an external module, in addition to changes to handle multi-channel images and uncertainty estimation using MC-Dropout. 

### Usage:

Follow these steps to use this model in your project:

* Clone this repository into a local folder
  `git clone https://github.com/kilgore92/PyTorch_ProbUNet.git`

* Install the model in your python environment
  `python setup.py install`

* Import the model into your script
  `from probabilistic_unet.model import ProbabilisticUnet`

If you have any questions, please open a pull-request or issue and I will get back to you.
