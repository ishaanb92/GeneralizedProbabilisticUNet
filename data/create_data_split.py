"""

Script to divide the LIDC dataset into train-val-test sets
and save each set as a dictionary

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

SEED=1234
DATA_DIR = '/data/ishaan/LIDC'
TRAIN_FRAC = 0.8

import random
import joblib
import os
from math import floor


def construct_data_dict(patient_ids, key_constructor_dict,full_data_dict):

    data_dict = {}
    for pat_id in patient_ids:
        per_patient_slice_key = key_constructor_dict[pat_id]
        for slice_key in per_patient_slice_key:
            data_key = '{}_{}'.format(pat_id, slice_key)
            data_dict[data_key] = full_data_dict[data_key]

    return data_dict


# Read the data dict
data_dict = joblib.load(os.path.join(DATA_DIR, 'data_lidc.pickle'))

data_dict_keys = sorted(data_dict.keys())

patient_ids = []

# Dictionary to store slices+lesion info to construct the full key
full_key_constructor_dict = {}
prev_patient_id = ''
unique_patients = 0

for key in data_dict_keys:
    patient_id = key.split('_')[0]


    if patient_id != prev_patient_id: # New patient => construct an empty list to store the slices and lesion IDs

        # The slices belonging to the same patient might not be saved consecutively.
        if patient_id not in full_key_constructor_dict:
            full_key_constructor_dict[patient_id] = []
            unique_patients += 1

        prev_patient_id = patient_id

    full_key_constructor_dict[patient_id].append(key.split('_')[1])

    patient_ids.append(patient_id)

# Get unique patient IDs
patient_ids = sorted(full_key_constructor_dict.keys())
num_patients = len(patient_ids)
print('Number of patients : {}'.format(num_patients))

# Same fraction of patients in validation and test sets
val_frac = (1-TRAIN_FRAC)/2

num_train_patients = floor(TRAIN_FRAC*num_patients + 0.5)
num_val_patients = floor(val_frac*num_patients + 0.5)

random.Random(SEED).shuffle(patient_ids)

train_patients = patient_ids[:num_train_patients]
val_patients = patient_ids[num_train_patients:(num_train_patients+num_val_patients)]
test_patients = patient_ids[(num_val_patients+num_train_patients):]

print('Number of training patients: {} validation patients: {} test patients: {}'.format(len(train_patients), len(val_patients), len(test_patients)))


# Construct data dict from the split keys!
train_data_dict = construct_data_dict(patient_ids=train_patients,
                                      key_constructor_dict=full_key_constructor_dict,
                                      full_data_dict=data_dict)


val_data_dict = construct_data_dict(patient_ids=val_patients,
                                    key_constructor_dict=full_key_constructor_dict,
                                    full_data_dict=data_dict)


test_data_dict = construct_data_dict(patient_ids=test_patients,
                                     key_constructor_dict=full_key_constructor_dict,
                                     full_data_dict=data_dict)

print('Training slices = {}, val slices = {}, test slices = {}'.format(len(train_data_dict.keys()),len(val_data_dict.keys()),len(test_data_dict.keys())))
print('Total number of slices = {}'.format(len(data_dict.keys())))

# Save each separate dict!
joblib.dump(train_data_dict, os.path.join(DATA_DIR, 'train_data.pkl'))
joblib.dump(val_data_dict, os.path.join(DATA_DIR, 'val_data.pkl'))
joblib.dump(test_data_dict, os.path.join(DATA_DIR, 'test_data.pkl'))
