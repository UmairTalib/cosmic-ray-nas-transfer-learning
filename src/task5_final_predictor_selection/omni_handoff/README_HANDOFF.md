# Omni-Cluster Handoff — NAS-Bench-201 Weights for RAML Task 5

**For:** cluster setup and download on the Omni cluster (Uni Siegen)
**Goal:** obtain the trained NAS-Bench-201 weights for **18 specific architectures**
and send back ~30 small weight files (a few tens of MB total).

---

## Why this is needed

Our RAML transfer-learning pipeline (task5) fine-tunes 30 networks (top-10
predicted per dataset) on the Cosmic ray dataset. The professor asked us to
initialise these from the **NAS-Bench-201 pretrained weights**.

Those weights only exist inside a **~360 GB split-tar** (`NATS-tss-v1_0-3ffb9-full`,
12 parts of ~30 GB). We cannot download it on the local RAML machine:
- the old 226 GB single-archive Google Drive link is **dead (404)**;
- the current 12-part folder is **quota-blocked for anonymous `gdown`** and is
  **larger than the free disk** on the local machine.

The Omni cluster has the bandwidth + storage to do this. **We only need 18 of the
15,625 files**, so after extraction you send back a tiny bundle.

---

## What to run (3 steps)

Everything runs on a **login / transfer node** (compute nodes usually have no
internet). Use a scratch/work filesystem with **≥ 750 GB free**.

```bash
# 0. Put this handoff folder on the cluster, then:
export DEST=/scratch/$USER/nats_full          # or any big-disk path
export HANDOFF_DIR=$PWD                        # dir containing these files

# 1. Download the 12 parts (~360 GB) + verify md5
bash 01_download.sh                            # METHOD=A (gdown) default
#   If gdown hits the quota error, switch to authenticated rclone:
#   METHOD=B REMOTE=gdrive bash 01_download.sh

# 2. Reassemble + extract ONLY the 18 needed files
bash 02_extract.sh                             # -> $DEST/needed_full/

# 3. Pull out the state_dicts -> 30 small .pt files
python 03_extract_state_dicts.py --full-dir $DEST/needed_full --out weights_out

# 4. Bundle and send back (this is the only thing we need!)
tar -czf raml_weights.tar.gz -C weights_out .
```

Send `raml_weights.tar.gz` back to us. It contains files named
`NNNNNN_<dataset>.pt` (e.g. `009448_cifar100.pt`) — one per fine-tuning run.

---

## The 18 architectures we need

Full list is in `needed_architectures.json` (machine-readable) and
`needed_indices.txt` (one 6-digit index per line). Each file
`NNNNNN.pickle.pbz2` inside the tar contains all datasets + seeds for that
architecture; we only extract the source datasets listed below.

| nats_idx | encoding | source dataset(s) needed |
|---------:|----------|--------------------------|
| 000137 | S-3-P-S-3-3 | cifar10 |
| 002352 | 3-1-3-P-3-1 | cifar100 |
| 002658 | 3-S-S-N-3-3 | cifar10, cifar100 |
| 002870 | 3-3-1-S-1-S | cifar100, ImageNet16-120 |
| 003275 | 3-3-N-1-3-S | ImageNet16-120 |
| 003990 | 3-1-3-3-N-1 | ImageNet16-120 |
| 005115 | 3-3-1-1-3-3 | ImageNet16-120 |
| 005204 | 1-S-3-S-1-3 | cifar10, ImageNet16-120 |
| 007335 | 3-3-1-3-1-1 | cifar100, ImageNet16-120 |
| 007632 | 3-3-3-3-3-1 | cifar100 |
| 008002 | 3-3-3-S-P-1 | cifar10 |
| 008993 | S-3-3-3-3-1 | cifar10, cifar100, ImageNet16-120 |
| 009448 | 1-S-S-3-3-3 | cifar10, cifar100, ImageNet16-120 |
| 010004 | N-3-S-3-P-3 | cifar10 |
| 010436 | 3-1-3-1-3-3 | cifar100, ImageNet16-120 |
| 014255 | 1-S-N-3-3-3 | cifar10, cifar100 |
| 014289 | 3-3-S-P-1-3 | cifar10, cifar100, ImageNet16-120 |
| 014410 | 3-N-N-1-3-N | cifar10 |

18 unique architectures → 30 (architecture, dataset) weight files.

---

## Download source (reference)

- Google Drive folder: `https://drive.google.com/drive/folders/17S2Xg_rVkUul4KuJdq0WaWoUuDbo8ZKB`
- Parts: `NATS-tss-v1_0-3ffb9-full.taraa` … `.taral` (12 files)
- `md5sum.txt` in the folder has the checksums (also fetched by `01_download.sh`)
- Official docs: https://github.com/D-X-Y/NATS-Bench

### If `gdown` keeps failing (quota)
Use **rclone** (authenticated — bypasses the anonymous quota):
1. On any machine with a browser: `rclone config` → new remote `gdrive`
   (type: Google Drive, default scopes).
2. In the browser, open the shared folder → **Add shortcut to My Drive**.
3. Copy `~/.config/rclone/rclone.conf` to the cluster.
4. `METHOD=B REMOTE=gdrive bash 01_download.sh`

---

## Notes / gotchas

- **Disk:** parts alone are ~360 GB. `02_extract.sh` streams the parts through
  `tar` (no second 360 GB copy). If you prefer concatenating first, ensure
  ~720 GB free (see Option 2 in the script).
- **No internet on compute nodes:** do the download on a login/transfer node;
  extraction can run in a batch job.
- **Sanity check naming:** if `02_extract.sh` finds 0 members, run
  `cat *.tara* | tar -tf - | head` to see the real internal path naming and
  adjust the grep in the script.
- We already have the **simple** archive (`NATS-tss-v1_0-3ffb9-simple`) locally;
  the full one is identical in structure but with `net_state_dict` populated.
