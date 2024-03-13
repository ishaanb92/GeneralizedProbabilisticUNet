"""
Script to test the Probabilistc U-Net for the QUBIC challenge validation data

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import torch
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import shutil
from utils.utils import *
from probabilistic_unet.model import ProbabilisticUnet
from probabilistic_unet.unet import Unet
from qubic_datasets.dataset import QubicDataset, QubicPancreas
from wmh_dataset.dataset import WMHDataset
from argparse import ArgumentParser
import sys
from lidc_dataset.dataset import LIDC
from random import sample, shuffle
import pickle
import pandas as pd
import os
from segmentation_metrics.metrics import dice_score
from skimage.transform import resize
import SimpleITK as sitk
import joblib
from helper_functions import *

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_SAMPLES = 10000



def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--test_data_dir', type=str, default='/home/ishaan/qubic/validation_data/validation_data_v2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--mc_dropout', action='store_true')
    parser.add_argument('--low_rank', action='store_true')
    parser.add_argument('--full_cov', action='store_true')
    parser.add_argument('--flow', action='store_true')
    parser.add_argument('--glow', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--use_tune', action='store_true')
    parser.add_argument('--mixture_model', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--model', type=str, default='punet')
    parser.add_argument('--zdim', type=int, default=-1)
    parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument('--n_components', type=int, default=-1)
    args = parser.parse_args()
    return args

def test(args):

    if args.gpu_id >= 0:
        device = 'cuda:{}'.format(args.gpu_id)
    else:
        device=  'cpu'



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Running on device {}'.format(device))

    if args.ensemble is False and args.use_tune is True:
        checkpoint_dir = os.path.join(args.checkpoint_dir, '{}'.format(args.dataset))
    else:
        checkpoint_dir = args.checkpoint_dir


    if args.dataset != 'lidc' and args.dataset != 'wmh':
        patient_dirs = [f.path for f in os.scandir(os.path.join(args.test_data_dir, '{}'.format(args.dataset), 'Validation')) if f.is_dir()]
        print('Number of patients in test set = {}'.format(len(patient_dirs)))

    num_samples = args.num_samples

    if args.dataset == 'brain-growth' or args.dataset == 'kidney' or args.dataset == 'lidc' or args.dataset == 'pancreatic-lesion' or args.dataset == 'pancreas':
        num_tasks = 1
        num_classes = 1
        in_channels = 1
        one_hot = True
    elif args.dataset == 'wmh':
        num_tasks = 1
        num_classes = 1
        in_channels = 2
        one_hot = True
    elif args.dataset == 'brain-tumor':
        num_tasks = 3
        num_classes = 3
        in_channels = 4
        one_hot = False
    elif args.dataset == 'prostate':
        num_tasks = 2
        num_classes = 2
        in_channels = 1
        one_hot = False

    if args.dataset != 'lidc' and args.dataset != 'wmh':
        if args.dataset != 'pancreatic-lesion' and args.dataset != 'pancreas':
            test_dataset = QubicDataset(data_dirs=patient_dirs,
                                        dataset=args.dataset,
                                        mode=args.mode)
        else:
            test_dataset = QubicPancreas(data_dirs=patient_dirs,
                                         dataset=args.dataset,
                                         mode=args.mode)

        n_slices = test_dataset.__len__()

        test_dataloader = DataLoader(dataset=test_dataset,
                                     num_workers=4,
                                     batch_size=args.batch_size,
                                     shuffle=False)
    elif args.dataset == 'lidc':
        dataset = LIDC(data_dir=args.test_data_dir)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)

        valid_indices, test_indices, train_indices = indices[:split], indices[split:2*split], indices[2*split:]

        n_slices = split

        # Ensure the order of slices processed is the same across all variants!
        generator = torch.Generator()
        generator.manual_seed(args.seed)

        test_sampler = SubsetRandomSampler(test_indices,
                                           generator=generator)

        test_dataloader = DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     sampler=test_sampler)
    elif args.dataset == 'wmh':

        test_dataset = WMHDataset(data_dir=args.test_data_dir,
                                  mode='test')

        n_slices = test_dataset.__len__()

        test_dataloader = DataLoader(dataset=test_dataset,
                                     num_workers=4,
                                     batch_size=args.batch_size,
                                     shuffle=False)

    models = []

    if args.model == 'punet':
        # Load and parse .pkl file with the optimal hyper-parameters
        if args.use_tune is True:
            try:
                config = joblib.load(os.path.join(args.checkpoint_dir, '{}'.format(args.dataset), 'best_config.pkl'))
            except FileNotFoundError:
                if args.dataset == 'lidc':
                    config = {}
                    config['unet_mode'] = 'shallow'
                    if args.debug is False:
                        config['z_dim'] = 6
                    else:
                        config['z_dim'] = 2
                    config['beta'] = 1.0

            if config is None and args.dataset == 'lidc':
                config = {}
                config['unet_mode'] = 'shallow'
                if args.debug is False:
                    config['z_dim'] = 6
                else:
                    config['z_dim'] = 2
                beta = 1.0

            print('Best config : {}'.format(config))

            if config['unet_mode'] == 'deep':
                num_filters = [32, 64, 128, 256, 512]
            elif config['unet_mode'] == 'shallow':
                num_filters = [32, 64, 128, 256]
            else:
                raise ValueError('{} is not a valid option for unet_mode'.format(config['unet_mode']))

            if args.low_rank is True:
                z_dim, rank = config['z_rank']
            else:
                rank=-1
                z_dim = config['z_dim']

            if args.mixture_model is True:
                n_components = config['n_components']
                temperature = config['temperature']
            else:
                n_components = 1
                temperature = 0.1

            beta = config['beta']
            checkpoint_dir = os.path.join(checkpoint_dir, 'best_trial', 'checkpoints')
        else:
            z_dim = args.zdim
            if args.low_rank is True:
                rank = args.rank
            if args.mixture_model is True:
                n_components = args.n_components
                temperature = args.temperature
            else:
                n_components = 1
                temperature = 1.0

            num_filters = [32, 64, 128, 256]
            checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
            beta = 1.0

        # Define the model
        model = ProbabilisticUnet(input_channels=in_channels,
                                  label_channels=num_classes,
                                  num_filters=num_filters,
                                  num_classes=num_classes,
                                  mc_dropout=False,
                                  dropout_rate=0.0,
                                  latent_dim=z_dim,
                                  beta=beta,
                                  no_convs_fcomb=4,
                                  low_rank=args.low_rank,
                                  rank=rank,
                                  full_cov=args.full_cov,
                                  n_components=n_components,
                                  temperature=temperature,
                                  norm=True,
                                  flow=args.flow,
                                  glow=args.glow)
        # Load the checkpoint
        load_dict = load_model(model=model,
                               checkpoint_dir=checkpoint_dir,
                               training=False)

        model = load_dict['model']

        models.append(model)

    elif args.model == 'unet':
        if args.ensemble is True:
            ckpt_seed_dirs = [os.path.join(f.path, '{}'.format(args.dataset)) for f in os.scandir(checkpoint_dir) if f.is_dir()]
            ckpt_seed_dirs_to_load = ckpt_seed_dirs[:num_samples]
        else:
            ckpt_seed_dirs_to_load = [checkpoint_dir]

        # Load the U-Net model(s)
        for ckpt_dir in ckpt_seed_dirs_to_load:
            config = joblib.load(os.path.join(ckpt_dir, 'best_config.pkl'))
            print('Best config : {}'.format(config))
            if config['unet_mode'] == 'deep':
                num_filters = [32, 64, 128, 256, 512]
            elif config['unet_mode'] == 'shallow':
                num_filters = [32, 64, 128, 256]
            else:
                raise ValueError('{} is not a valid option for unet_mode'.format(config['unet_mode']))

            if args.mc_dropout is True:
                dropout_rate = config['dropout_rate']
            else:
                dropout_rate = 0.0

            model = Unet(input_channels=in_channels,
                         num_classes=num_classes,
                         num_filters=num_filters,
                         initializers={'w':'he_normal', 'b':'normal'},
                         apply_last_layer=True,
                         norm=True,
                         mc_dropout=args.mc_dropout,
                         dropout_rate=dropout_rate)

            # Load the checkpoint
            load_dict = load_model(model=model,
                                   checkpoint_dir=os.path.join(ckpt_dir, 'best_trial', 'checkpoints'),
                                   training=False)

            model = load_dict['model']

            models.append(model)


    for model in models:
        model.eval()
        enable_dropout(model)

    dice_scores = []

    ged_scores = []

    sample_diversity = []

    io_variability_list = []

    patient_ids = []

    mixture_entropy = []

    distribution_entropy = []

    avg_pairwise_dist = []

    avg_pairwise_iou_list = []

#    ncc_list = []

    ece_list = []

    print('Number of slices in test-set: {}'.format(n_slices))

    with torch.no_grad():
        for idx, data_dict in enumerate(test_dataloader):

            patch = data_dict['images'].float()
            mask = data_dict['labels'].float()

            if num_classes == 1:
                mask = torch.unsqueeze(mask, dim=2)

            if args.dataset != 'lidc' and args.dataset != 'pancreatic-lesion' and args.dataset != 'pancreas' and args.dataset != 'wmh':
                case_dirs = data_dict['case_dir']

            patch = patch.to(device)

            h, w = patch.shape[-2:]

            samples = torch.zeros((num_samples, patch.shape[0] , num_classes, h, w),
                                  dtype=torch.float32,
                                  device='cpu')

            if args.ensemble is False:
                model = models[-1]
                model.to(device)

            if args.model == 'punet':
                model.forward(patch=patch,
                              segm=None,
                              training=False)

            for sample_idx in range(num_samples):
                if args.ensemble is True:
                    model = models[sample_idx]
                    model.to(device)

                if args.model == 'punet':
                    punet_sample, _  = model.sample(testing=False)
                elif args.model == 'unet':
                    punet_sample = model.forward(patch)

                punet_sample = torch.sigmoid(punet_sample)

                samples[sample_idx, ...] = punet_sample.cpu()

            # Make batch-size the first axis
            samples = samples.permute((1, 0, 2, 3, 4))

            # Binarize the predictions
            bin_samples = (samples>torch.Tensor([0.5])).float()*1

            mask = mask.cpu()

            # Compute GED
            ged_score = np.zeros((patch.shape[0], num_classes), dtype=np.float32)
            sample_diversity_score = np.zeros((patch.shape[0], num_classes), dtype=np.float32)
            io_variability_dist = np.zeros((patch.shape[0], num_classes), dtype=np.float32)
            avg_pairwise_iou = np.zeros((patch.shape[0], num_classes), dtype=np.float32)
#            ncc = np.zeros((patch.shape[0], num_classes), dtype=np.float32)
            ece = np.zeros((patch.shape[0], num_classes), dtype=np.float32)

            for class_id in range(num_classes):
                for batch_idx in range(patch.shape[0]):
                    score_dict = compute_ged_metric(gt=mask[batch_idx, :, class_id, ...],
                                                    seg=bin_samples[batch_idx, :, class_id, ...])

                    ged_score[batch_idx, class_id] = score_dict['GED']
                    sample_diversity_score[batch_idx, class_id] = score_dict['Sample Diversity']
                    io_variability_dist[batch_idx, class_id] = score_dict['IO Variability']

                    avg_pairwise_iou[batch_idx, class_id] = compute_avg_pairwise_iou(gt=mask[batch_idx, :, class_id, ...],
                                                                                     seg=samples[batch_idx, :, class_id, ...])

#                    ncc[batch_idx, class_id] = compute_ncc(gt=mask[batch_idx, :, class_id, ...],
#                                                           seg=samples[batch_idx, :, class_id, ...])


                    ece[batch_idx, class_id] = compute_ece(gt=mask[batch_idx, :, class_id, ...],
                                                           seg=samples[batch_idx, :, class_id, ...])

            ged_scores.extend(list(ged_score))
            sample_diversity.extend(list(sample_diversity_score))
            io_variability_list.extend(list(io_variability_dist))
            avg_pairwise_iou_list.extend(list(avg_pairwise_iou))
#            ncc_list.extend(list(ncc))
            ece_list.extend(list(ece))

            # Save the batch of images, predictions and mask
            p_unet_samples = maybe_convert_tensor_to_numpy(bin_samples)
            mean_prediction = maybe_convert_tensor_to_numpy(torch.mean(samples, dim=1))
            mean_mask = maybe_convert_tensor_to_numpy(torch.mean(mask, dim=1))

            # Compute Dice
            dice_score = np.zeros((patch.shape[0], num_classes), dtype=np.float32)
            for class_id in range(num_classes):
                dice_score[:, class_id] = compute_dice_score(gt=mean_mask[:, class_id, ...],
                                                             seg=mean_prediction[:, class_id, ...])

            dice_scores.extend(dice_score)

            if args.model == 'punet':
                prior_dist = model.get_latent_space_distribution(mode='prior')

                z_samples_matrix = generate_samples_matrix(dist=prior_dist,
                                                           n_samples=N_SAMPLES,
                                                           show_indices=args.mixture_model)



                z_samples_matrix = maybe_convert_tensor_to_numpy(z_samples_matrix)

                # Compute distribution entropy
                dist_entropy = compute_distribution_entropy(prior_dist=prior_dist,
                                                            mixture_model=args.mixture_model)

                distribution_entropy.extend(list(dist_entropy))


                if args.mixture_model is True: # Get categorical distributions prob
                    cat_dist = prior_dist.get_mixture_distribution()
                    comp_dist = prior_dist.get_component_distribution()

                    # Compute average pair-wise distance between component distributions
                    avg_pairwise_dist_batch = compute_avg_pairwise_distance(comp_dist,
                                                                            low_rank=args.low_rank)

                    avg_pairwise_dist.extend(list(avg_pairwise_dist_batch))

                    probs = cat_dist.probs
                    probs = maybe_convert_tensor_to_numpy(probs)
                    entropy_batch = compute_entropy(probs)
                    mixture_entropy.extend(list(entropy_batch))
                else:
                    probs = None
            else:
                z_samples_matrix = None


            if args.dataset == 'lidc' or args.dataset == 'pancreatic-lesion' or args.dataset == 'pancreas' or args.dataset == 'wmh':
                save_outputs_lidc(image=maybe_convert_tensor_to_numpy(patch),
                                  prediction=p_unet_samples,
                                  label=maybe_convert_tensor_to_numpy(mask),
                                  save_dir=args.save_dir,
                                  batch_idx=idx,
                                  z_samples=z_samples_matrix)
            else:
                save_outputs(images=maybe_convert_tensor_to_numpy(patch),
                             predictions=p_unet_samples,
                             labels=mask,
                             save_dir=args.save_dir,
                             one_hot=one_hot,
                             case_dirs=case_dirs,
                             dataset=args.dataset,
                             n_tasks=num_tasks,
                             z_samples=z_samples_matrix,
                             probs=probs)

    # Save the metrics
    if args.ensemble is False and args.use_tune is True:
        result_dir = os.path.join(args.checkpoint_dir, '{}'.format(args.dataset))
    else:
        result_dir = args.checkpoint_dir

    result_dict = {}

    # Plot entropy
    if args.mixture_model is True:
        plot_hist(array=mixture_entropy,
                  vert_value=np.log(n_components),
                  fname=os.path.join(args.save_dir, 'mixture_entropy.png'))

        plot_hist(array=avg_pairwise_dist,
                  vert_value=None,
                  xlabel='Jensen-Shannon divergence',
                  fname=os.path.join(args.save_dir, 'average_pairwise_distance.png'))

    if one_hot is True:
        dice_scores = np.array(dice_scores)
        ged_scores = np.array(ged_scores)
        sample_diversity = np.array(sample_diversity)
        io_variability_list = np.array(io_variability_list)
        avg_pairwise_iou_list = np.array(avg_pairwise_iou_list)
#        ncc_list = np.array(ncc_list)
        ece_list = np.array(ece_list)

        if args.model == 'punet':
            distribution_entropy = np.array(distribution_entropy)

        result_dict['Dice'] = dice_scores[:, 0]
        result_dict['GED'] = ged_scores[:, 0]
        result_dict['Sample Diversity'] = sample_diversity[:, 0]
        result_dict['IO Variability'] = io_variability_list[:, 0]
        result_dict['Avg. IoU'] = avg_pairwise_iou_list[:, 0]
#        result_dict['NCC'] = ncc_list[:, 0]
        result_dict['ECE'] = ece_list[:, 0]

        if args.model == 'punet':
            result_dict['Distribution Entropy'] = distribution_entropy
    else:
        dice_scores = np.array(dice_scores)
        ged_scores = np.array(ged_scores)
        for class_id in range(num_classes):
            result_dict['Dice Task {}'.format(class_id)] = dice_scores[:, class_id]
            result_dict['GED Task {}'.format(class_id)] = ged_scores[:, class_id]
            result_dict['Sample Diversity {}'.format(class_id)] = sample_diversity[:, class_id]

    result_df = pd.DataFrame.from_dict(data=result_dict)

    if args.dataset == 'lidc' or args.dataset == 'pancreatic-lesion' or args.dataset == 'pancreas' or args.dataset == 'wmh':
        print('GED:: {} +/- {}'.format(result_df['GED'].mean(), result_df['GED'].std()))
        print('Sample Diversity:: {} +/- {}'.format(result_df['Sample Diversity'].mean(), result_df['Sample Diversity'].std()))
        print('Inter-observer variability :: {} +/- {}'.format(result_df['IO Variability'].mean(), result_df['IO Variability'].std()))
        print('Avg. pairwise IoU :: {} +/- {}'.format(result_df['Avg. IoU'].mean(), result_df['Avg. IoU'].std()))
        print('ECE :: {} +/- {}'.format(result_df['ECE'].mean(), result_df['ECE'].std()))
    else:
        print(result_df)

    result_df.to_pickle(os.path.join(result_dir, 'metrics_{}_samples.pkl'.format(args.num_samples)))



#

if __name__ == '__main__':
    args = build_parser()
    test(args)
