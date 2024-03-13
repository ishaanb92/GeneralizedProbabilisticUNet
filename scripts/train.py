"""
Script to train the Probabilistc U-Net for the QUBIC challenge data

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import shutil
from utils.utils import *
from qubic_datasets.dataset import QubicDataset, QubicPancreas
from lidc_dataset.dataset import LIDC
from wmh_dataset.dataset import WMHDataset
from probabilistic_unet.model import ProbabilisticUnet
from probabilistic_unet.loss import *
from probabilistic_unet.unet import Unet
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import random
from dice_loss import binary_dice_loss
import pandas as pd
from torchearlystopping.pytorchtools import EarlyStopping
import os
from math import floor, isnan
import joblib
from helper_functions import *
import numpy as np
# Hyper-parameter optimization
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='/home/ishaan/qubic/training_data/training_data_v2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default='punet')
    parser.add_argument('--checkpoint_dir', type=str, default='/data/ishaan/qubic/ray_trials')
    parser.add_argument('--config_dir', type=str, default=None)
    parser.add_argument('--mc_dropout', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--flow', action='store_true')
    parser.add_argument('--glow', action='store_true')
    parser.add_argument('--low_rank', action='store_true')
    parser.add_argument('--full_cov', action='store_true')
    parser.add_argument('--mixture_model', action='store_true')
    parser.add_argument('--use_tune', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_trials', type=int, default=50)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--grace_period', type=int, default=100)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--zdim', type=int, default=6)
    parser.add_argument('--num_raters', type=int, default=-1)
    parser.add_argument('--train_seed', type=int, default=-1)
    parser.add_argument('--ensemble', action='store_true')
    args = parser.parse_args()
    return args


def train(config, checkpoint_dir=None, train_data_dir=None, dataset=None, model_str='punet', mc_dropout=False, batch_size=6, seed=42, use_tune=True, low_rank=False, mixture_model=False, patience=100, flow=False, glow=False, full_cov=False, ensemble=False, train_seed=-1, num_raters=-1):


    if torch.cuda.is_available():
        device='cuda:0'
    else:
        raise RuntimeError('GPU not available!')

    # Set the seed -- Used for data splitting (esp. LIDC)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if use_tune is True:
        trial_dir = tune.get_trial_dir()
        log_dir = os.path.join(trial_dir, 'logs')
        os.makedirs(log_dir)
        ckpt_dir = os.path.join(trial_dir, 'checkpoints')
        os.makedirs(ckpt_dir)
        print('Logs directory created at {}'.format(os.path.join(trial_dir, 'logs')))
    else:
        log_dir = os.path.join(checkpoint_dir, 'logs')
        ckpt_dir = os.path.join(checkpoint_dir, 'checkpoints')
        os.makedirs(log_dir)
        os.makedirs(ckpt_dir)
        print('Checkpoint directory: {}'.format(ckpt_dir))
        print('Logs directory: {}'.format(log_dir))
        print('Training config: {}'.format(config))

    if dataset == 'brain-tumor':
        num_tasks = 3
        num_classes = 3
        in_channels = 4
        one_hot = False
        loss_mode = 'bce'
    elif dataset == 'prostate':
        num_tasks = 2
        num_classes = 2
        in_channels = 1
        one_hot = False
        loss_mode = 'bce'
    elif dataset == 'wmh':
        num_tasks = 1
        num_classes = 1
        in_channels = 2
        one_hot = True
        loss_mode = 'ce'
    else: # Default settings
        num_tasks = 1
        num_classes = 1
        in_channels = 1
        one_hot = True
        loss_mode = 'ce'

    if config['unet_mode'] == 'deep':
        num_filters = [32, 64, 128, 256, 512]
    elif config['unet_mode'] == 'shallow':
        num_filters = [32, 64, 128, 256]
    else:
        raise ValueError('{} is not a valid option for unet_mode'.format(config['unet_mode']))

    if model_str == 'punet':
        if low_rank is True:
            z_dim, rank = config['z_rank']
        else:
            rank = -1
            z_dim = config['z_dim']

        if mixture_model is True:
            n_components = config['n_components']
            temperature = config['temperature']
            gamma = 0.0
        else:
            n_components = 1
            temperature = 0.1
            gamma = 0.0

    if mc_dropout is False:
        dropout_rate = 0.0
    else:
        dropout_rate = config['dropout_rate']



    if dataset != 'lidc' and dataset != 'wmh':
        # Split the training set into an 80-20 train/val split
        patient_dirs = [f.path for f in os.scandir(os.path.join(train_data_dir, '{}'.format(dataset), 'Training')) if f.is_dir()]
        n_training_patients = floor(0.8*len(patient_dirs) + 0.5)
        n_val_patients = len(patient_dirs) - n_training_patients
        # Shuffle the list
        random.Random(seed).shuffle(patient_dirs)

        train_dirs = patient_dirs[:n_training_patients]
        val_dirs = patient_dirs[n_training_patients:]

        print('Number of patients in training data = {}'.format(len(train_dirs)))
        print('Number of patients in validation data = {}'.format(len(val_dirs)))

    # FIXME
#    if args.renew is True or os.path.exists(checkpoint_dir) is False:
#        try:
#            shutil.rmtree(checkpoint_dir)
#            print('Deleting old checkpoint directory')
#        except FileNotFoundError:
#            pass
#        os.makedirs(checkpoint_dir)
#        os.makedirs(log_dir)
#
#        n_iter = 0
#        n_iter_val = 0
#        epoch_saved = -1
#
#        # Split the training set into an 80-20 train/val split
#        patient_dirs = [f.path for f in os.scandir(os.path.join(train_data_dir, '{}'.format(dataset), 'Training')) if f.is_dir()]
#        n_training_patients = floor(0.8*len(patient_dirs) + 0.5)
#        n_val_patients = len(patient_dirs) - n_training_patients
#        # Shuffle the list
#        random.Random(args.seed).shuffle(patient_dirs)
#
#        train_dirs = patient_dirs[:n_training_patients]
#        val_dirs = patient_dirs[n_training_patients:]
#
#        # Save these directories
#        joblib.dump(train_dirs, os.path.join(checkpoint_dir, 'train_dirs.pkl'))
#        joblib.dump(val_dirs, os.path.join(checkpoint_dir, 'val_dirs.pkl'))
#
#    else: # Load model and continue training
#        load_dict = load_model(model=model,
#                               optimizer=optimizer,
#                               checkpoint_dir=checkpoint_dir,
#                               training=True)
#
#        n_iter = load_dict['n_iter']
#        n_iter_val = load_dict['n_iter_val']
#        optimizer = load_dict['optimizer']
#        model = load_dict['model']
#        epoch_saved = load_dict['epoch']
#
#        # Load the saved train and val directories
#        train_dirs = joblib.load(os.path.join(checkpoint_dir, 'train_dirs.pkl'))
#        val_dirs = joblib.load(os.path.join(checkpoint_dir, 'val_dirs.pkl'))
#



    # Create dataset class and dataloader
    if dataset != 'lidc' and dataset != 'wmh':

        if dataset != 'pancreas' and dataset != 'pancreatic-lesion':
            train_dataset = QubicDataset(data_dirs=train_dirs,
                                         dataset=dataset,
                                         mode='train')

            val_dataset = QubicDataset(data_dirs=val_dirs,
                                       dataset=dataset,
                                       mode='val')
        else:
            train_dataset = QubicPancreas(data_dirs=train_dirs,
                                          dataset=dataset,
                                          mode='train')

            val_dataset = QubicPancreas(data_dirs=val_dirs,
                                        dataset=dataset,
                                        mode='val')


        train_dataloader = DataLoader(dataset=train_dataset,
                                      num_workers=4,
                                      batch_size=batch_size,
                                      shuffle=True)

        val_dataloader = DataLoader(dataset=val_dataset,
                                    num_workers=4,
                                    batch_size=batch_size,
                                    shuffle=False)
    elif dataset == 'lidc':
        dataset = LIDC(data_dir=train_data_dir)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)

        valid_indices, test_indices, train_indices = indices[:split], indices[split:2*split], indices[2*split:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    elif dataset == 'wmh':

        train_dataset = WMHDataset(data_dir=train_data_dir,
                                   mode='train')

        val_dataset = WMHDataset(data_dir=train_data_dir,
                                 mode='val')

        train_dataloader = DataLoader(dataset=train_dataset,
                                      num_workers=4,
                                      batch_size=batch_size,
                                      shuffle=True)

        val_dataloader = DataLoader(dataset=val_dataset,
                                    num_workers=4,
                                    batch_size=batch_size,
                                    shuffle=False)

    # Manages early stopping within a trial
    early_stopper = EarlyStopping(patience=patience,
                                  checkpoint_dir=ckpt_dir,
                                  delta=0.0)

    # For ensembles use a different seed to acheive
    # different initialization of the model
    if train_seed > 0:
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)


    # Define model and optimizer
    if model_str == 'punet':

        model = ProbabilisticUnet(input_channels=in_channels,
                                  num_classes=num_classes,
                                  label_channels=num_classes,
                                  num_filters=num_filters,
                                  latent_dim=z_dim,
                                  no_convs_fcomb=4,
                                  beta=config['beta'],
                                  gamma=gamma,
                                  norm=True,
                                  low_rank=low_rank,
                                  rank=rank,
                                  n_components=n_components,
                                  temperature=temperature,
                                  flow=flow,
                                  glow=glow,
                                  full_cov=full_cov)

    elif model_str == 'unet':
        model = Unet(input_channels=in_channels,
                     num_classes=num_classes,
                     num_filters=num_filters,
                     initializers={'w':'he_normal', 'b':'normal'},
                     apply_last_layer=True,
                     norm=True,
                     mc_dropout=mc_dropout,
                     dropout_rate=dropout_rate)


    else:
        raise ValueError('Unsupported model {}'.format(model_str))

    optimizer = optim.Adam(model.parameters(),
                           lr=config['lr'],
                           weight_decay=1e-5)

    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)
    model.train()

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    n_iter = 0
    n_iter_val = 0
    epoch_saved = -1
    prev_best_epoch = -1

    if dataset == 'pancreatic-lesion':
        fg_weight = torch.Tensor([5.0]).float()
    else:
        fg_weight = None


    for epoch in range(0, 10000):
        for idx, data_dict in enumerate(train_dataloader):

            images = data_dict['images'].float()
            labels = data_dict['labels'].float()

            # Restrict available training labels to num_raters
            if num_raters > 0:
                if dataset == 'lidc':
                    labels = labels[:, :num_raters, ...]
                elif dataset == 'wmh':
                    labels = labels[:, -num_raters:, ...] # 1 raters => ground truth (best)

            model.train()

            optimizer.zero_grad()
            model.zero_grad()

            random_label_idx = torch.randperm(n=labels.shape[1])[0]
            loss_label = labels[:, [random_label_idx]]

            if num_tasks > 1:
                loss_label = torch.flatten(loss_label, start_dim=1, end_dim=2)

            if model_str == 'punet':

                try:
                    model.forward(images.to(device),
                                  loss_label.to(device),
                                  training=True)

                except RuntimeError as e:
                    print('exiting with error : {}'.format(e))
                    return

                _,recon_loss,kl_loss,elbo = model.elbo(loss_label.to(device),
                                                       use_mask=False,
                                                       analytic_kl=not(mixture_model),
                                                       pos_weight=fg_weight,
                                                       mc_samples=1000)

                reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior)
                loss = -elbo + 1e-5 * reg_loss

            elif model_str == 'unet':
                pred = model.forward(images.to(device))

                loss = F.binary_cross_entropy_with_logits(input=pred,
                                                          target=loss_label.to(device),
                                                          reduction='none')

                loss = torch.sum(loss)
                loss = loss/batch_size

            loss.backward()
            optimizer.step()

            # Log the losses
            writer.add_scalar('train/loss', loss.item(), n_iter)
            if model_str == 'punet':
                writer.add_scalar('train/reconstruction_loss', recon_loss.item(), n_iter)
                writer.add_scalar('train/KL', kl_loss.item(), n_iter)
                # Plot weights of the mixture weights
                if mixture_model is True:
                    prior_dist = model.get_latent_space_distribution(mode='prior')
                    posterior_dist = model.get_latent_space_distribution(mode='posterior')
                    prior_mixture_distribution = prior_dist.get_mixture_distribution()
                    posterior_mixture_distribution = posterior_dist.get_mixture_distribution()
                    prior_mix_weights = prior_mixture_distribution.probs
                    posterior_mix_weights = posterior_mixture_distribution.probs
                    # Mean over the batch
                    batch_mean_prior_mix_weights = torch.mean(prior_mix_weights, dim=0)
                    batch_mean_posterior_mix_weights = torch.mean(posterior_mix_weights, dim=0)
                    for mix_component in range(n_components):
                        writer.add_scalar('train/prior_mix_weight_{}'.format(mix_component),
                                          batch_mean_prior_mix_weights[mix_component].item(),
                                          n_iter)


                        writer.add_scalar('train/posterior_mix_weight_{}'.format(mix_component),
                                          batch_mean_posterior_mix_weights[mix_component].item(),
                                          n_iter)

            n_iter+=1

        # End of epoch
        print('Epoch {} ends'.format(epoch))

        with torch.no_grad():
            # Start validation loop
            val_loss = []
            model.apply(set_mcd_eval_mode)
            for idx, data_dict in enumerate(val_dataloader):

                images = data_dict['images'].float()
                labels = data_dict['labels'].float()

                # Restrict available labels to num_raters
                if num_raters > 0:
                    if dataset == 'lidc':
                        labels = labels[:, :num_raters, ...]
                    elif dataset == 'wmh':
                        labels = labels[:, -num_raters:, ...] # 1 raters => ground truth (best)

                random_label_idx = torch.randperm(n=labels.shape[1])[0]
                loss_label = labels[:, [random_label_idx]]

                if num_tasks > 1:
                    loss_label = torch.flatten(loss_label, start_dim=1, end_dim=2)

                if model_str == 'punet':
                    try:
                        model.forward(images.to(device),
                                      loss_label.to(device),
                                      training=True)

                    except RuntimeError as e:
                        print('exiting with error : {}'.format(e))
                        return

                    _,recon_loss,kl_loss,elbo = model.elbo(loss_label.to(device),
                                                           use_mask=False,
                                                           analytic_kl=not(mixture_model),
                                                           pos_weight=fg_weight,
                                                           mc_samples=1000)

                    reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior)
                    loss = -elbo + 1e-5 * reg_loss

                    val_loss.append(loss.item())

                elif model_str == 'unet':
                    random_label_idx = np.random.randint(low=0, high=labels.shape[1])
                    pred = model.forward(images.to(device))

                    loss = F.binary_cross_entropy_with_logits(input=pred,
                                                              target=loss_label.to(device),
                                                              reduction='none')
                    loss = torch.sum(loss)
                    loss = loss/batch_size

                    val_loss.append(loss.item())

                writer.add_scalar('val/loss', val_loss[-1], n_iter_val)
                if model_str == 'punet':
                    writer.add_scalar('val/reconstruction_loss', recon_loss.item(), n_iter_val)
                    writer.add_scalar('val/KL', kl_loss.item(), n_iter_val)
                    if mixture_model is True:
                        prior_dist = model.get_latent_space_distribution(mode='prior')
                        posterior_dist = model.get_latent_space_distribution(mode='posterior')
                        prior_mixture_distribution = prior_dist.get_mixture_distribution()
                        posterior_mixture_distribution = posterior_dist.get_mixture_distribution()
                        prior_mix_weights = prior_mixture_distribution.probs
                        posterior_mix_weights = posterior_mixture_distribution.probs
                        # Mean over the batch
                        batch_mean_prior_mix_weights = torch.mean(prior_mix_weights, dim=0)
                        batch_mean_posterior_mix_weights = torch.mean(posterior_mix_weights, dim=0)
                        for mix_component in range(n_components):
                            writer.add_scalar('val/prior_mix_weight_{}'.format(mix_component),
                                              batch_mean_prior_mix_weights[mix_component].item(),
                                              n_iter_val)


                            writer.add_scalar('val/posterior_mix_weight_{}'.format(mix_component),
                                              batch_mean_posterior_mix_weights[mix_component].item(),
                                              n_iter_val)
                n_iter_val += 1

            # Compute mean validation loss
            mean_val_loss = np.mean(np.array(val_loss))

            if isnan(mean_val_loss):
                print('See a NaN validation loss value, exiting!')
                return

            if mean_val_loss < 0 and gamma == 0:
                print('Mean validation loss is negative {}. Exiting'.format(mean_val_loss))
                return

            print('Epoch {} :: Mean validation loss = {}'.format(epoch, mean_val_loss))
            writer.add_scalar('val/mean_loss', mean_val_loss, epoch)

            # Save checkpoint only if validation loss improves
            early_stop, best_epoch = early_stopper(val_loss=mean_val_loss,
                                                   curr_epoch=epoch,
                                                   model=model,
                                                   optimizer=optimizer,
                                                   scheduler=None,
                                                   scaler=None,
                                                   n_iter=n_iter,
                                                   n_iter_val=n_iter_val)

            if use_tune is True:
                # FIXED: Reporting the validation loss to tune ONLY if it improves
                # renders the scheduler useless because it will never terminate
                # "bad" trials. So we report the mean validation loss for every epoch
                # but save the checkpoint only when it improves. When comparing trials
                # we use the scope='all' so the best trial is not chosen based on the
                # results from last epoch but across all epochs!
                tune.report(mean_val_loss=mean_val_loss)

            # If we hit early stopping condition for this trial, stop it
            if early_stop is True:
                print('Early stopping condition for this trial reached with best epoch {}'.format(best_epoch))
                return



if __name__ == '__main__':

    args = build_parser()

    # Configure GPU visibility (at the top!)
    assert(len(args.gpus) <= 4)

    gpu_str = ''

    for idx, gpu in enumerate(args.gpus):
        gpu_str += '{}'.format(gpu)
        if idx != len(args.gpus)-1:
            gpu_str += ','

    print('GPU string: {}'.format(gpu_str))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

    if args.use_tune is True:
        assert(torch.cuda.device_count() == len(args.gpus))

    # Create the dataset-specific checkpoint directory
    local_dir = os.path.join(args.checkpoint_dir, '{}'.format(args.dataset))

    # Case-1: Checkpoint directory for the dataset exists
    if os.path.exists(local_dir) is True:
        if args.use_tune is True: # Case-1a: We want to delete the old results and perform hyper-param optimization again
            print('Deleting old raytune logs')
            shutil.rmtree(local_dir)
        else: # Case-1b: We want to delete the "best trial" and re-train using the optimal hyper-parameters (This is required if there is a discrepancy between the 'best' checkpoint and the 'best' hyper-parameters)
            if os.path.exists(os.path.join(local_dir, 'best_config.pkl')) is True:
                print('Removing previous best trial directory')
                try:
                    shutil.rmtree(os.path.join(local_dir, 'best_trial'))
                except FileNotFoundError:
                    pass
            else:
                raise ValueError('Optimal hyper-parameter configuration not found in {}'.format(local_dir))

            config = joblib.load(os.path.join(local_dir, 'best_config.pkl'))
            print('Start training with hyper-parameters : {}'.format(config))

    # Case-2: Checkpoint directory for the dataset does not exists
    else:
        if args.use_tune is False: # Case-2a: We want to use hyper-parameters from a completed run (eg: Ensembles)
            if args.config_dir is not None:
                config = joblib.load(os.path.join(args.config_dir, '{}'.format(args.dataset),'best_config.pkl'))
                print('Start training with hyper-parameters : {}'.format(config))
            else:
                print('Start training with user-supplied hyper-params')
                config = {}
                config ['lr'] = args.lr
                config['unet_mode'] = 'shallow'

                if args.model != 'unet':
                    config['beta'] = 1.0
                    config['z_dim'] = args.zdim
                    # TODO: Fix hyper-params for other variants (rank, n_comp etc.)

        else: # Case-2b: We want to start hyper-parameter optimization for the first time!
            print('Start hyper-parameter optimization')
            pass


    if args.use_tune is True:
        scheduler = ASHAScheduler(max_t=10000,
                                  grace_period=args.grace_period,
                                  reduction_factor=2)
        config = {'lr':1e-4}

        # Probabilistic U-Net params: Beta (KL-weight) and latent space dimensions
        if args.model != 'unet':
            config['beta'] = 1.0

            # For low-rank, the rank and z-dim are dependendent because rank is allowed to only take values from [1, z-dim-1]
            if args.low_rank is False:
                if args.debug is False:
                    config['z_dim'] = tune.grid_search([2, 4, 6, 8, 16, 32])
                else:
                    config['z_dim'] = 2

        if args.mc_dropout is True or args.low_rank is True or args.mixture_model is True or args.full_cov is True or args.ensemble is True:

            if args.dataset != 'lidc':
                assert(args.config_dir is not None) # We need architectural hyper-params from the baseline config to isolate the effects of our additions
                config_probunet = joblib.load(os.path.join(args.config_dir, '{}'.format(args.dataset),'best_config.pkl'))
                # Fixed hyper-parameters (related to the NN architecture)
                # We vary hyper-parameters related to training dynamics (eg: LR, KL-weight (beta))
                unet_mode = config_probunet['unet_mode']
                config['unet_mode'] = unet_mode
            else:
                config['unet_mode'] = "shallow"

            # Dropout parameters
            if args.mc_dropout is True:
                config['dropout_rate'] = tune.grid_search([0.1, 0.3, 0.5])

            if args.low_rank is True or args.mixture_model is True:
                assert(args.model != 'unet') # U-Net does not support these configurations!

            # Low-rank parameters (rank)
            if args.low_rank is True:
                # See: https://docs.ray.io/en/latest/tune/faq.html#conditional-spaces
                def _iter_z_rank():
                    for z in [2, 4, 6, 8]:
                        for rank in range(1, z):
                            yield z, rank

                config['z_rank'] = tune.grid_search(list(_iter_z_rank()))

            # Mixture model parameters (n_components and temperature for GS distribution)
            if args.mixture_model is True:
                config['n_components'] = tune.randint(lower=2, upper=11) #  2 <= n_components <= 10
                config['temperature'] = tune.uniform(0.1, 0.5)

        else: # Try different depth configs for basic PUNet
            config['unet_mode'] = "shallow"

        # FIXME: Currently hyper-param optimization works can only use RandomSearch
        result = tune.run(tune.with_parameters(train,
                                               dataset=args.dataset,
                                               model_str=args.model.lower(),
                                               train_data_dir=args.train_data_dir,
                                               batch_size=args.batch_size,
                                               mc_dropout=args.mc_dropout,
                                               seed=args.seed,
                                               use_tune=args.use_tune,
                                               low_rank=args.low_rank,
                                               mixture_model=args.mixture_model,
                                               patience=args.patience,
                                               flow=args.flow,
                                               glow=args.glow,
                                               full_cov=args.full_cov,
                                               ensemble=args.ensemble,
                                               train_seed=args.train_seed,
                                               num_raters=args.num_raters),
                          resources_per_trial={'gpu':1},
                          config=config,
                          metric='mean_val_loss',
                          mode='min',
                          scheduler=scheduler,
                          num_samples=args.num_trials,
                          local_dir=local_dir,
                          keep_checkpoints_num=1,
                          raise_on_failed_trial=False)

        # We set scope='all' so that the trial with overall min. val loss is chosen
        # This is in line with how we checkpoint the model
        best_config = result.get_best_config(metric='mean_val_loss',
                                             mode='min',
                                             scope='all')

        best_logdir =  result.get_best_logdir(metric='mean_val_loss',
                                              mode='min',
                                              scope='all')


        joblib.dump(best_config, os.path.join(local_dir, 'best_config.pkl'))

        # Clean-up : delete all sub-optimal trials so that we don't run out of disk space
        clean_up_ray_trials(exp_dir=local_dir,
                            best_trial_dir=best_logdir,
                            new_dir=os.path.join(local_dir, 'best_trial'))

    else: # No hyper-parameter tuning because best hyper-parameters have been found
        # DEBUG
        train(config=config,
              checkpoint_dir=os.path.join(local_dir, 'best_trial'),
              dataset=args.dataset,
              train_data_dir=args.train_data_dir,
              batch_size=args.batch_size,
              mc_dropout=args.mc_dropout,
              seed=args.seed,
              use_tune=False,
              low_rank=args.low_rank,
              mixture_model=args.mixture_model,
              patience=args.patience,
              flow=args.flow,
              glow=args.glow,
              num_raters=args.num_raters)

        joblib.dump(config, os.path.join(local_dir, 'best_config.pkl'))
