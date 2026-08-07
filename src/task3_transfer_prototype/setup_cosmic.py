"""
One-time setup: generate train_dirs.npy and test_dirs.npy for the cosmic dataset,
and run a quick sanity check on the data.

Run from RAML/:
    python task3/setup_cosmic.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'task3'))

from cosmic_data import build_dir_lists
import numpy as np
import torch

COSMIC_ROOT = 'Cosmic'

print('Generating directory lists ...')
build_dir_lists(COSMIC_ROOT)

# Load and inspect
train_dirs = np.load(os.path.join(COSMIC_ROOT, 'train_dirs.npy'), allow_pickle=True)
test_dirs  = np.load(os.path.join(COSMIC_ROOT, 'test_dirs.npy'),  allow_pickle=True)
print(f'\nSummary:')
print(f'  train files: {len(train_dirs)}')
print(f'  test  files: {len(test_dirs)}')

# Quick data check: load one sample and verify shapes
if len(train_dirs) > 0:
    print('\nLoading one training sample ...')
    data = np.load(train_dirs[0])
    print(f'  raw npy shape: {data.shape}  dtype: {data.dtype}')
    print(f'  data[0] (image) range: [{data[0].min():.3f}, {data[0].max():.3f}]')
    print(f'  data[1] (mask)  unique values: {np.unique(data[1][:10,:10])}')
    if data.shape[0] == 3:
        print(f'  data[2] (ignore) unique: {np.unique(data[2])}')

    from cosmic_data import PairedDatasetImagePath
    ds = PairedDatasetImagePath(train_dirs[:5])
    img, mask = ds[0]
    print(f'\nDataset item 0:')
    print(f'  image shape: {tuple(img.shape)}  dtype: {img.dtype}')
    print(f'  mask  shape: {tuple(mask.shape)}  dtype: {mask.dtype}')
    print(f'  mask  unique values: {mask.unique().tolist()}')
    print('\nSetup complete. Ready to run finetune_cosmic.py')
