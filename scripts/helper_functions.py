"""

Script for misc. functions that are shared

"""
import os
import shutil
from distutils.dir_util import copy_tree
import numpy as np
import torch
from segmentation_metrics.metrics import *
import SimpleITK as sitk
import pandas as pd
import joblib
import torch.nn as nn
import sys
sys.path.append(os.path.join(os.getcwd(), 'analysis'))
from analysis_utils import *
import scipy
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal, Independent, kl, LowRankMultivariateNormal, Categorical, MixtureSameFamily

EPS = 1e-6

def clean_up_ray_trials(exp_dir=None,
                        best_trial_dir=None,
                        new_dir=None):
    """
    Function to move the "winning" checkpoint dir to a separate folder while retaining the rest
    for analysis later

    """
    assert(exp_dir is not None)
    assert(best_trial_dir is not None)

    # Using shutil.move() function to change the checkpoint directory caused
    # a corruption in the file due to which the model could not be loaded
    # during inference. This has been fixed by use the copy_tree function
    # from the distutils package
    if new_dir is not None: # Copy the best trial directory to the new place
        if os.path.exists(new_dir) is False:
            os.makedirs(new_dir)
        copy_tree(os.path.join(best_trial_dir, 'checkpoints'), os.path.join(new_dir, 'checkpoints'))
        copy_tree(os.path.join(best_trial_dir, 'logs'), os.path.join(new_dir, 'logs'))


def maybe_convert_tensor_to_numpy(data):

    if isinstance(data, torch.Tensor) is True:
        if data.device != 'cpu':
            data = data.cpu()

        data = data.numpy().astype(np.float32)

    return data


def l2_regularisation(m):
    l2_reg = None
    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg



def compute_entropy(probs):

    # Probs : [B, n_components]
    batchSz, n_components = probs.shape
    entropy = -1*np.sum(np.multiply(probs, np.log(probs)), axis=1)

    return entropy

def save_outputs(images=None, predictions=None, labels=None, save_dir=None, one_hot=False, case_dirs=None, dataset=None, n_tasks=-1, z_samples=None, probs=None):


    images = maybe_convert_tensor_to_numpy(images)
    predictions = maybe_convert_tensor_to_numpy(predictions)
    labels = maybe_convert_tensor_to_numpy(labels)

    viz_dir = os.path.join(save_dir, dataset)

    if os.path.exists(viz_dir) is False:
        os.makedirs(viz_dir)

    if probs is not None: # Probs shape: [B, n_components]
        # Compute entropy
        batchSz, n_components = probs.shape
        entropy = -1*np.sum(np.multiply(probs, np.log(probs)), axis=1)

        # Each Gaussian in the mixture will be chosen with equal probability
        max_entropy = -np.log(1/n_components)

        plot_hist(array=entropy,
                  vert_value=max_entropy,
                  fname=os.path.join(viz_dir, 'mixture_entropy.png'))

    for patient_id, case_dir in enumerate(case_dirs):

        pat_dir = os.path.join(viz_dir, case_dir)
        if os.path.exists(pat_dir) is False:
            os.makedirs(pat_dir)

        # Save the samples
        if z_samples is not None:
            z_sample = z_samples[:, patient_id, ...]
            np.save(file=os.path.join(pat_dir, 'prior_samples.npy'), arr=z_sample)
            plot_2d_scatterplot(matrix=z_sample,
                                fname=os.path.join(pat_dir, 'prior_sample_plot.png'),
                                )
        image = images[patient_id, ...]
        prediction = predictions[patient_id, ...]
        label = labels[patient_id, ...]

        mean_labels = np.mean(label, axis=0)
        mean_prediction = np.mean(prediction, axis=0)

        # Save the prediction
        if one_hot is True:
            save_image(np.squeeze(mean_prediction, axis=0), fname=os.path.join(pat_dir, 'mean_pred_task01.nii.gz'))
            save_image(np.squeeze(prediction, axis=1), fname=os.path.join(pat_dir, 'samples_pred_task01.nii.gz'))
            save_image(np.squeeze(mean_labels, axis=0), fname=os.path.join(pat_dir, 'mean_label_task01.nii.gz'))
            save_image(np.squeeze(label, axis=1), fname=os.path.join(pat_dir, 'samples_label_task01.nii.gz'))
        else:
            prediction_series = []
            label_series = []
            save_image(mean_prediction, fname=os.path.join(pat_dir, 'mean_pred.nii.gz'))
            save_image(mean_labels, fname=os.path.join(pat_dir, 'mean_label.nii.gz'))

            for task_id in range(n_tasks):
                prediction_series.append(sitk.GetImageFromArray(prediction[:, task_id, ...]))
                label_series.append(sitk.GetImageFromArray(label[:, task_id, ...]))

            prediction_series_itk = sitk.JoinSeries(prediction_series)
            label_series_itk = sitk.JoinSeries(label_series)

            sitk.WriteImage(prediction_series_itk, os.path.join(pat_dir, 'samples_pred.nii.gz'))
            sitk.WriteImage(label_series_itk, os.path.join(pat_dir, 'samples_label.nii.gz'))

        # Save the image
        save_image(image, fname=os.path.join(pat_dir, 'image.nii.gz'))


def save_image(image, fname=None):

    assert(isinstance(image, np.ndarray))
    image_itk = sitk.GetImageFromArray(image)

    sitk.WriteImage(image_itk, fname)



def compute_dice_score(seg=None, gt=None):

    gt = maybe_convert_tensor_to_numpy(gt)
    seg = maybe_convert_tensor_to_numpy(seg)

    # Binarize both arrays
    if gt.dtype != np.uint8:
        gt = np.where(gt>0.5, 1, 0).astype(np.uint8)
    if seg.dtype != np.uint8:
        seg = np.where(seg>0.5, 1, 0).astype(np.uint8)

    return dice_score(gt=gt, seg=seg)


# pdist and compute_ged_metric are taken from
# https://github.com/raghavian/cFlow/blob/master/utils/utils.py#L80
def pdist(a,b):
    N = a.shape[1]
    M = b.shape[1]
    H = a.shape[-2]
    W = a.shape[-1]
#    C = a.shape[2]

    aRep = a.repeat(1,M,1,1).view(-1,N,M,H,W)
    bRep = b.repeat(1,N,1,1).view(-1,M,N,H,W).transpose(1,2)

#    print('N : {}'.format(N))
#    print('M : {}'.format(M))
#    print('A : {}'.format(aRep.shape))
#    print('B : {}'.format(bRep.shape))

    inter = (aRep & bRep).float().sum(-1).sum(-1) + EPS
    union = (aRep | bRep).float().sum(-1).sum(-1) + EPS

    IoU = inter/union
    dis = (1-IoU).mean(-1).mean(-1)
    return dis

#def compute_ged_metric(seg,gt):
#
#    if isinstance(seg, np.ndarray):
#        seg = torch.Tensor(seg)
#    if isinstance(gt, np.ndarray):
#        gt = torch.Tensor(gt)
#
#    seg = seg.type(torch.ByteTensor)
#    gt = gt.type_as(seg)
#
#    dSP = pdist(gt,seg)
#    dSS = pdist(gt,gt)
#    dPP = pdist(seg, seg)
#
#    metric_dict = {}
#    metric_dict['GED'] = (2*dSP - dSS - dPP).numpy()
#    metric_dict['Sample Diversity'] = dPP.numpy()
#    metric_dict['Pred-Label distance'] = dSP.numpy()
#    metric_dict['IO Variability'] = dSS.numpy()
#
#    return metric_dict


def compute_ged_metric(seg=None, gt=None):

    gt = maybe_convert_tensor_to_numpy(gt)
    seg = maybe_convert_tensor_to_numpy(seg)

    # Binarize both arrays
    if gt.dtype != np.uint8:
        gt = np.where(gt>0.5, 1, 0).astype(np.uint8)
    if seg.dtype != np.uint8:
        seg = np.where(seg>0.5, 1, 0).astype(np.uint8)

    return compute_ged(seg=seg, gt=gt)



#def compute_qubic_metric(pred=None, seg=None, one_hot=True):
#
#    if pred.device != 'cpu':
#        pred = pred.cpu()
#
#    if seg.device != 'cpu':
#        seg = seg.cpu()
#
#    # B x 2 x H x W (one_hot = True)
#    # B x N_TASKS x H x W (one_hot = False)
#    pred = pred.numpy()
#    # B x N_ANNO x 1 x 2 x H x W (one_hot = True)
#    # B x N_ANNO x N_TASKS x H x W (one_hot = False)
#    seg = seg.numpy()
#
#    n_tasks = seg.shape[2]
#
#    # Mean over annotations
#    # B x 1 x 2 x H x W (one_hot = True)
#    # B x N_TASKS x H x W (one_hot = False)
#    mean_seg = np.mean(seg, axis=1)
#
#    if one_hot is True:
#        # B x 2 x H x W
#        dice_scores = []
#        mean_seg = np.squeeze(mean_seg, axis=1)
#        for thresh in thresholds:
#            thresh_seg = np.where(mean_seg > thresh, 1, 0).astype(np.float32)
#            thresh_pred = np.where(pred > thresh, 1, 0).astype(np.float32)
#            dice_scores.append(dice_score(thresh_pred[:, 1, ...], thresh_seg[:, 1, ...]))
#
#        mean_dice = np.mean(np.array(dice_scores))
#
#        print('Task 1 dice score : {}'.format(mean_dice))
#    else:
#        for task_id in range(n_tasks):
#            dice_scores = []
#            task_seg = mean_seg[:, task_id, ...]
#            task_pred = pred[:, task_id, ...]
#            for thresh in thresholds:
#                thresh_seg = np.where(task_seg > thresh, 1, 0).astype(np.float32)
#                thresh_pred = np.where(task_pred > thresh, 1, 0).astype(np.float32)
#                dice_scores.append(dice_score(thresh_pred, thresh_seg))
#
#            mean_dice = np.mean(np.array(dice_scores))
#
#            print('Task {} dice score : {}'.format(task_id+1, mean_dice))
#
#    return mean_dice
#

def save_outputs_pancreas(image=None, prediction=None, mask=None, checkpoint_dir=None, case_dir=None, dataset=None):

    # Create the directory
    save_dir = os.path.join(checkpoint_dir, 'test_results', dataset, case_dir)

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    mean_pred = np.mean(prediction, axis=0)
    mean_mask = np.mean(mask, axis=0)

    mean_pred_itk = sitk.GetImageFromArray(mean_pred)
    image_itk = sitk.GetImageFromArray(image)
    mean_mask_itk = sitk.GetImageFromArray(mean_mask)

    pred_itk = create_4d_image_itk(prediction)
    mask_itk = create_4d_image_itk(mask)

    # Save all the images
    sitk.WriteImage(image_itk, os.path.join(save_dir, 'image.nii.gz'))
    sitk.WriteImage(mean_pred_itk, os.path.join(save_dir, 'mean_prediction.nii.gz'))
    sitk.WriteImage(mean_mask_itk, os.path.join(save_dir, 'mean_label.nii.gz'))
    sitk.WriteImage(pred_itk, os.path.join(save_dir, 'prediction.nii.gz'))
    sitk.WriteImage(mask_itk, os.path.join(save_dir, 'label.nii.gz'))

def create_4d_image_itk(image_np):
    """
    Create 4-D ITK image, assuming the channel-first axis arrangements

    """
    n_channels = image_np.shape[0]

    image_series = []

    for channel in range(n_channels):
        image_series.append(sitk.GetImageFromArray(image_np[channel, ...]))

    image_series_itk = sitk.JoinSeries(image_series)

    return image_series_itk

def save_outputs_lidc(image=None, prediction=None, label=None, save_dir=None, batch_idx=-1, z_samples=None):
    assert(batch_idx >= 0)


    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    n_channels = image.shape[1]

    # Get rid of fake task axis
    prediction = np.squeeze(prediction, axis=2)
    label = np.squeeze(label, axis=2)

    # Image shape : [B, 1, 128, 128]
    if n_channels ==  1:
        image = np.squeeze(image, axis=1)
        image_itk = sitk.GetImageFromArray(image)
    else:
        image_itk = create_4d_image_itk(image)

    # Prediction shape : [B, N_SAMPLES, 128, 128]
    mean_prediction = np.mean(prediction, axis=1)
    prediction_itk = create_4d_image_itk(prediction)
    mean_prediction_itk = sitk.GetImageFromArray(mean_prediction)

    # Labels shape: [B, N_ANNO, 128, 128]
    mean_label = np.mean(label, axis=1)
    label_itk = create_4d_image_itk(label)
    mean_label_itk = sitk.GetImageFromArray(mean_label)

    sitk.WriteImage(image_itk, os.path.join(save_dir, 'image_batch_{}.nii.gz'.format(batch_idx)))
    sitk.WriteImage(mean_prediction_itk, os.path.join(save_dir, 'mean_prediction_batch_{}.nii.gz'.format(batch_idx)))
    sitk.WriteImage(mean_label_itk, os.path.join(save_dir, 'mean_label_batch_{}.nii.gz'.format(batch_idx)))
    sitk.WriteImage(prediction_itk, os.path.join(save_dir, 'prediction_batch_{}.nii.gz'.format(batch_idx)))
    sitk.WriteImage(label_itk, os.path.join(save_dir, 'label_batch_{}.nii.gz'.format(batch_idx)))

    if z_samples is not None:
        np.save(os.path.join(save_dir, 'z_samples_batch_{}.npy'.format(batch_idx)), z_samples)


def set_mcd_eval_mode(m):
    if type(m) == nn.Dropout or type(m) == nn.Dropout2d:
        m.train()
    else:
        m.eval()


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def calculate_iqr(x):
    assert(isinstance(x, np.ndarray))
    iqr = scipy.stats.iqr(x=x)
    return iqr


def compute_average_distribution(p, q, low_rank=False):
    """
    Create the M = (P + Q)/2 using a mixture of Gaussians

    """
    batch_shape = p.batch_shape[0]

    mix_probs = torch.zeros((batch_shape, 2))
    mix_probs[:, 0] = 0.5
    mix_probs[:, 1] = 0.5
    m_mix_dist = Categorical(probs=mix_probs)

    if low_rank is True:
        loc_p = torch.unsqueeze(p.loc, dim=1)
        loc_q = torch.unsqueeze(q.loc, dim=1)
        loc_m = torch.cat([loc_p, loc_q], dim=1)

        cov_factor_p = torch.unsqueeze(p.cov_factor, dim=1)
        cov_factor_q = torch.unsqueeze(q.cov_factor, dim=1)
        cov_factor_m = torch.cat([cov_factor_p, cov_factor_q], dim=1)

        cov_diag_p = torch.unsqueeze(p.cov_diag, dim=1)
        cov_diag_q = torch.unsqueeze(q.cov_dia, dim=1)
        cov_diag_m = torch.cat([cov_diag_p, cov_diag_q], dim=1)

        m_comp_dist = LowRankMultivariateNormal(loc=loc_m,
                                                cov_factor=cov_factor_m,
                                                cov_diag=cov_diag_m)


    else:
        loc_p = torch.unsqueeze(p.mean, dim=1)
        loc_q = torch.unsqueeze(q.mean, dim=1)
        loc_m = torch.cat([loc_p, loc_q], dim=1)

        scale_p = torch.unsqueeze(p.variance, dim=1)
        scale_q = torch.unsqueeze(q.variance, dim=1)
        scale_m = torch.cat([scale_p, scale_q], dim=1)

        m_comp_dist = Independent(Normal(loc=loc_m,
                                         scale=scale_m), 1)


    m_dist = MixtureSameFamily(m_mix_dist,
                               m_comp_dist)

    return m_dist



def compute_kl(p, q, mc_samples=1000):
    p_samples = p.sample(sample_shape=torch.Size([mc_samples]))

    p_logprob = p.log_prob(p_samples)
    q_logprob = q.log_prob(p_samples)

    kl_div = torch.mean((p_logprob - q_logprob), dim=0)

    return kl_div

def compute_jsd(p, q):
    """
    Compute the Jensen-Shannon divergence between pair of distributions

    """
    m = compute_average_distribution(p, q) # M = (P + Q)/2

    p_m_kl = compute_kl(p, m)
    q_m_kl = compute_kl(q, m)

    jsd = 0.5*(p_m_kl + q_m_kl)

    return jsd


def compute_avg_pairwise_distance(comp_dist, low_rank=False):
    """
    Compute average pairwise distance between different component distributions
    in a mixture family

    """
    batch_shape = comp_dist.batch_shape[0]

    distributions = []

    if low_rank is True:
        locs = comp_dist.loc.cpu()
        cov_factors = comp_dist.cov_factor.cpu()
        cov_diags = comp_dist.cov_diag.cpu()
        n_comps = locs.shape[1]

        for comp_idx in range(n_comps):
            comp_loc = locs[:, comp_idx, :]
            comp_cov_factor = cov_factors[:, comp_idx, :]
            comp_cov_diag = cov_diags[:, comp_idx, :]

            distributions.append(LowRankMultivariateNormal(loc=comp_loc,
                                                           cov_factor=comp_cov_factor,
                                                           cov_diag=comp_cov_diag))
    else:
        locs = comp_dist.mean.cpu()
        scales = comp_dist.variance.cpu()
        n_comps = locs.shape[1]

        for comp_idx in range(n_comps):
            comp_loc = locs[:, comp_idx, :]
            comp_scale = scales[:, comp_idx, :]

            distributions.append(Independent(Normal(loc=comp_loc,
                                                    scale=comp_scale), 1))

    # Compute avg. pairwise distances
    low_tri_elems = (n_comps*(n_comps-1))/2
    pairwise_distance = torch.zeros(size=(n_comps, n_comps, batch_shape),
                                    dtype=torch.float32)
    for i in range(n_comps):
        for j in range(i+1, n_comps):
            pairwise_distance[i, j, :] = compute_jsd(distributions[i], distributions[j])


    avg_pairwise_distance = torch.sum(pairwise_distance, dim=(0,1))/low_tri_elems

    return maybe_convert_tensor_to_numpy(avg_pairwise_distance)


def compute_avg_pairwise_iou(seg, gt):

    # Convert to numpy
    seg = maybe_convert_tensor_to_numpy(seg)
    gt = maybe_convert_tensor_to_numpy(gt)

    # Binarize
    seg = np.where(seg>0.5, 1, 0).astype(np.uint8)
    gt = np.where(gt>0.5, 1, 0).astype(np.uint8)

    avg_iou = compute_pairwise_iou(seg, gt)

    return avg_iou

def compute_distribution_entropy(prior_dist=None, mixture_model=False):

    # Since mixture models do not have a closed-form expression for
    # distribution entropy, we use an MC approximation
    if mixture_model is True:
        z_samples = prior_dist.rsample(sample_shape=torch.Size([10000]))
        log_prior_prob = prior_dist.log_prob(z_samples)
        dist_entropy = -1*torch.mean(log_prior_prob, dim=0)
    else:
        dist_entropy = prior_dist.entropy()

    dist_entropy = maybe_convert_tensor_to_numpy(dist_entropy)

    return dist_entropy


def compute_ncc(seg, gt):


    # Convert to numpy
    seg = maybe_convert_tensor_to_numpy(seg)
    gt = maybe_convert_tensor_to_numpy(gt)

    ncc = variance_ncc_dist(sample_arr=seg,
                            gt_arr=gt)

    return ncc


def compute_ece(seg, gt):

    seg = maybe_convert_tensor_to_numpy(seg)
    gt = maybe_convert_tensor_to_numpy(gt)

    # Compute mean over predictions
    seg = np.mean(seg, axis=0)
    gt = np.mean(gt, axis=0)

    # Binarize the ground truth!
    gt = np.where(gt > 0.5, 1.0, 0.0).astype(np.uint8)

    ece = expected_calibration_error(probabilities=seg,
                                     ground_truth=gt,
                                     bins=10)

    return ece
