"""
Step 3 — Pull the trained weights (net_state_dict) out of the 18 full NATS-tss
files and save one small .pt per (architecture, source-dataset) run.

Only these tiny files (~tens of MB total for all 30 runs) need to be sent back
to the RAML machine — NOT the 360 GB archive.

Usage:
    python 03_extract_state_dicts.py --full-dir /path/to/needed_full \
                                     --manifest needed_architectures.json \
                                     --out weights_out

The full NATS-tss pickle has the same layout as the simple version we already
use, except `net_state_dict` is populated:
    p['200']['all_results'][(dataset_key, seed)]['net_state_dict']
"""
import os, sys, bz2, pickle, argparse, json

import torch  # only needed to torch.save the state dicts

# source_dataset -> candidate NATS result keys (train+valid split preferred,
# then the *-valid split); seeds tried in order.
DATASET_KEYS = {
    'cifar10':        ['cifar10', 'cifar10-valid'],
    'cifar100':       ['cifar100', 'cifar100-valid'],
    'ImageNet16-120': ['ImageNet16-120', 'ImageNet16-120-valid'],
}
SEEDS = [777, 888, 999, 111, 0]

def find_state_dict(results, source_dataset):
    for dkey in DATASET_KEYS[source_dataset]:
        for seed in SEEDS:
            entry = results.get((dkey, seed))
            if entry and entry.get('net_state_dict') is not None:
                return entry['net_state_dict'], (dkey, seed)
    # last resort: any key matching the dataset with a non-null state dict
    for (dkey, seed), entry in results.items():
        if source_dataset.split('-')[0] in dkey and entry.get('net_state_dict') is not None:
            return entry['net_state_dict'], (dkey, seed)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full-dir', required=True,
                    help='dir with the 18 extracted NNNNNN.pickle.pbz2 files')
    ap.add_argument('--manifest', default=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'needed_architectures.json'))
    ap.add_argument('--out', default='weights_out')
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    os.makedirs(args.out, exist_ok=True)

    ok, missing = 0, []
    for a in manifest['architectures']:
        idx = a['nats_idx']
        # try both flat and nested layouts
        cand = [os.path.join(args.full_dir, f'{idx:06d}.pickle.pbz2')]
        for root, _, files in os.walk(args.full_dir):
            if f'{idx:06d}.pickle.pbz2' in files:
                cand.append(os.path.join(root, f'{idx:06d}.pickle.pbz2'))
        path = next((c for c in cand if os.path.exists(c)), None)
        if path is None:
            print(f'[{idx:06d}] FILE NOT FOUND'); missing.append((idx, 'file'))
            continue

        with bz2.BZ2File(path, 'rb') as f:
            p = pickle.load(f)
        results = p['200']['all_results']

        for ds in a['datasets']:
            sd, used = find_state_dict(results, ds)
            if sd is None:
                print(f'[{idx:06d}] {ds:15s} NO WEIGHTS in file'); missing.append((idx, ds))
                continue
            out = os.path.join(args.out, f'{idx:06d}_{ds}.pt')
            torch.save(sd, out)
            print(f'[{idx:06d}] {ds:15s} -> {os.path.basename(out)}  (from {used})')
            ok += 1

    print(f'\nSaved {ok}/30 weight files into {args.out}/')
    if missing:
        print('MISSING:', missing)
    print('\nZip and send back:  tar -czf raml_weights.tar.gz -C', args.out, '.')

if __name__ == '__main__':
    main()
