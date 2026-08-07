"""
Cosmic dataset utilities — adapted from:
https://github.com/linusericsson/einspace/blob/main/einspace/data_utils/cosmic.py

Only the parts needed for NAS-Bench-201 fine-tuning are kept.
Key facts:
  - Binary pixel segmentation: detect cosmic rays in HST images
  - Input : (batch, 1, 128, 128)  single-channel astronomy image
  - Target: (batch, 1, 128, 128)  binary mask; pixels == IGNORE_FLAG are excluded
  - Loss  : CosmicBCEWithLogitsLoss  (BCEWithLogits, masking ignore pixels)
  - Metric: AUROC pixel-wise         (higher = better)
"""

import os
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, Subset
from sklearn import metrics

IGNORE_FLAG = -9999

# Hubble field -> observation type; only GAL fields are used for training/test
FIELD_TYPE = {
    '10595_2':'GC','10595_7':'GC','9442_1':'GC','9442_3':'GC','9442_5':'GC',
    '10760_2':'GAL','10760_4':'GAL','10631_3':'EX','10631_1':'EX','10631_4':'EX',
    '12103_a3':'EX','13498_b1':'EX','13737_2':'GAL','13737_3':'GAL','9490_a3':'GAL',
    '10349_30':'GC','10005_10':'GC','10120_3':'GC','12513_2':'GAL','12513_3':'GAL',
    '14164_9':'EX','13718_6':'EX','10524_7':'GC','10182_pb':'GAL','10182_pd':'GAL',
    '9425_2':'EX','9425_4':'EX','9583_99':'EX','10584_13':'GAL','9978_5e':'EX',
    '15100_2':'EX','15647_13':'EX','11340_11':'GC','13389_10':'EX','9694_6':'EX',
    '10342_3':'GAL','14343_1':'GAL','10536_13':'EX','13057_1':'GAL','10260_7':'GAL',
    '10260_5':'GAL','10407_3':'GAL','13375_4':'EX','13375_7':'EX','13364_95':'GAL',
    '10190_28':'GAL','10190_13':'GAL','10146_4':'GC','10146_3':'GC','10775_ab':'GC',
    '11586_5':'GC','12438_1':'EX','13671_35':'EX','14164_1':'GC','9490_a2':'GAL',
    '9405_6d':'EX','9405_4b':'EX','9450_14':'EX','10092_1':'EX','13691_11':'GAL',
    '12058_12':'GAL','12058_16':'GAL','12058_1':'GAL','9450_16':'EX','10775_52':'GC',
    '12602_1':'GC','12602_2':'GC','10775_29':'GC','10775_ad':'GC',
    '12058_6':'GAL','14704_1':'GAL','13804_6':'GAL',
}


def build_dir_lists(cosmic_root):
    """
    Walk npy_train/ and npy_test/ inside cosmic_root and save path lists.
    Only f435w filter and GAL field type are used (matches original paper).
    Saves train_dirs.npy and test_dirs.npy inside cosmic_root.
    """
    for split in ('train', 'test'):
        base = os.path.join(cosmic_root, f'npy_{split}')
        paths = []
        f435_dir = os.path.join(base, 'f435w')
        if not os.path.isdir(f435_dir):
            print(f'WARNING: {f435_dir} not found — skipping {split}')
            continue
        for prop_id in os.listdir(f435_dir):
            prop_dir = os.path.join(f435_dir, prop_id)
            if not os.path.isdir(prop_dir):
                continue
            for vis_num in os.listdir(prop_dir):
                vis_dir = os.path.join(prop_dir, vis_num)
                if not os.path.isdir(vis_dir):
                    continue
                key = f'{prop_id}_{vis_num}'
                if FIELD_TYPE.get(key) != 'GAL':
                    continue
                for fname in os.listdir(vis_dir):
                    if fname.endswith('.npy') and fname != 'sky.npy':
                        paths.append(os.path.join(vis_dir, fname))
        out_path = os.path.join(cosmic_root, f'{split}_dirs.npy')
        np.save(out_path, np.array(paths))
        print(f'  {split}: {len(paths)} files -> {out_path}')


def set_cosmic_input(img, mask, shape=128):
    img  = torch.tensor(img).view((-1, shape, shape))
    mask = torch.tensor(mask).view((-1, shape, shape))
    return img, mask


def get_fixed_crop(images, crop_height, crop_width):
    x, y = 64, 64
    return [img[y:y+crop_height, x:x+crop_width] for img in images]


class PairedDatasetImagePath(Dataset):
    """Load (image, mask) pairs from .npy files for deepCR cosmic dataset."""

    def __init__(self, paths, skyaug_min=0, skyaug_max=0):
        self.paths = np.array(paths)
        self.skyaug_min = skyaug_min
        self.skyaug_max = skyaug_max

    def __len__(self):
        return len(self.paths)

    def _get_skyaug(self, i):
        sky_path = os.path.join(os.path.split(self.paths[i])[0], 'sky.npy')
        if os.path.isfile(sky_path):
            f_img  = os.path.basename(self.paths[i])
            sky_idx = int(f_img.split('_')[0])
            sky = np.load(sky_path)[sky_idx - 1]
            return sky * np.random.uniform(self.skyaug_min, self.skyaug_max)
        return 0.0

    def __getitem__(self, i):
        data   = np.load(self.paths[i])
        image  = data[0]
        mask   = data[1]
        ignore = data[2] if data.shape[0] == 3 else np.zeros_like(data[0])

        skyaug = self._get_skyaug(i)
        image, mask, ignore = get_fixed_crop([image, mask, ignore], 128, 128)
        mask[ignore == 1] = mask[ignore == 1] + IGNORE_FLAG

        image, mask = set_cosmic_input(image + skyaug, mask, shape=128)
        return image.to(dtype=torch.float32), mask


def build_cosmic_datasets(cosmic_root, val_size=483, aug_sky=(-0.9, 3.0)):
    """
    Returns (trainset, valset, testset) ready for DataLoader.
    Expects train_dirs.npy and test_dirs.npy inside cosmic_root
    (generate with build_dir_lists() if not present).
    """
    train_dirs = np.load(os.path.join(cosmic_root, 'train_dirs.npy'), allow_pickle=True)
    test_dirs  = np.load(os.path.join(cosmic_root, 'test_dirs.npy'),  allow_pickle=True)

    full_train = PairedDatasetImagePath(train_dirs, aug_sky[0], aug_sky[1])
    testset    = PairedDatasetImagePath(test_dirs,  aug_sky[0], aug_sky[1])

    n = len(full_train)
    trainset = Subset(full_train, np.arange(n - val_size))
    valset   = Subset(full_train, np.arange(n - val_size, n))

    return trainset, valset, testset


class CosmicBCEWithLogitsLoss(BCEWithLogitsLoss):
    """BCEWithLogitsLoss that ignores pixels flagged with IGNORE_FLAG (target < 0)."""

    def forward(self, input, target):
        valid_mask   = target >= 0
        input_valid  = input[valid_mask]
        target_valid = target[valid_mask]
        if input_valid.numel() == 0:
            return input.sum() * 0.0
        return super().forward(input_valid, target_valid.float())


class CosmicMetricFunction:
    """Pixel-wise AUROC over the full dataset, ignoring IGNORE_FLAG pixels."""

    def __call__(self, preds, gts):
        # preds and gts are lists of numpy arrays
        pred_flat = np.concatenate([np.array(p).reshape(-1) for p in preds])
        gt_flat   = np.concatenate([np.array(g).reshape(-1) for g in gts])

        # Filter out ignore-flagged pixels
        valid = gt_flat >= 0
        pred_flat = pred_flat[valid]
        gt_flat   = gt_flat[valid].astype(np.int32)

        return metrics.roc_auc_score(gt_flat, pred_flat)
