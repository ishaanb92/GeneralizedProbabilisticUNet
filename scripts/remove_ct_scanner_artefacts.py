"""

Script to get rid of CT scanner artefacts (at the bottom)

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import numpy as np
import os
import SimpleITK as sitk




def clean_slice(image_slice):

    assert(image_slice.ndim == 2)

    # Clip values
    image_slice = np.clip(image_slice, a_min=-100, a_max=200)

    return image_slice


def remove_ct_scanner_artefacts(image_np):

    image_clean = np.zeros_like(image_np)

    n_slices, h, w = image_np.shape

    for slice_id in range(n_slices):
        image_clean[slice_id, :, :] = clean_slice(image_np[slice_id, :, :])

    return image_clean



if __name__ == '__main__':

    image_path = '/data/ishaan/qubic/training_data/training_data_v2/pancreas/case1-1/'

    # Read image
    image_itk = sitk.ReadImage(os.path.join(image_path, 'image.nii.gz'))

    image_np = sitk.GetArrayFromImage(image_itk)

    image_np = np.transpose(image_np, (2, 1, 0))

    image_clean = remove_ct_scanner_artefacts(image_np)

    image_clean_itk = sitk.GetImageFromArray(image_clean)

    sitk.WriteImage(image_clean_itk, os.path.join(image_path, 'image_clean.nii.gz'))

