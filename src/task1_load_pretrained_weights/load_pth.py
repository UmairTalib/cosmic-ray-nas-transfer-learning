"""
TODO 1 - Load .pth weights for DarcyFlow / NinaPro networks.

Key findings from inspecting all 20 .pth files
------------------------------------------------
Dataset    | #files | stem channels | spatial input | reduction-cell format
-----------|--------|---------------|---------------|----------------------
darcyflow  |   10   |       3       |   85 x 85     | OLD (layers.N.* keys)
ninapro    |   10   |       1       |   16 x 52     | NEW (conv_a/conv_b keys)

Two problems had to be solved:

  Problem 1 - Architecture mismatch for 85x85 inputs
      Current XAutoDL's ResNetBasicblock uses AvgPool2d(stride=2) for
      downsampling.  For 85x85:  AvgPool gives 42x42, while the stride-2
      3x3 conv gives 43x43.  Adding them causes a RuntimeError.
      The old architecture used matched 1x1 stride-2 convs on both paths,
      so spatial sizes always agreed.  Fixed by LegacyResNetBasicblock.

  Problem 2 - State-dict key mismatch
      The old XAutoDL reduction cells used layers.N.* keys (multiple
      sub-variants exist across files).  New XAutoDL uses conv_a/conv_b/
      downsample.  Ninapro files use the new keys; darcyflow use the old.
      Fixed per file by detecting the format and choosing the right loader.

  Problem 3 - Input channels
      XAutoDL's TinyNetwork hardcodes stem input as 3 channels.
      Ninapro stem has 1 channel (EMG signal).  Fixed by patching the stem
      after construction, before the state-dict load.
"""

import os, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from xautodl.models import get_cell_based_tiny_net
from xautodl.models.cell_operations import ReLUConvBN
from xautodl.models.cell_infers.cells import InferCell


# ── Legacy reduction-cell components (darcyflow OLD format) ──────────────────

class _PreprocessWrapper(nn.Module):
    """Creates  .preprocess.op.X  keys matching the OLD-A reduction cell."""
    def __init__(self, C_in, C_out, stride, affine=True):
        super().__init__()
        self.preprocess = ReLUConvBN(C_in, C_out, 1, stride, 0, 1, affine)

    def forward(self, x):
        return self.preprocess(x)


class _ConvBNShortcut(nn.Module):
    """Creates  .conv  /  .bn  keys for the stride-2 shortcut path."""
    def __init__(self, C_in, C_out, stride, affine=True):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride,
                              padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class LegacyResNetBasicblock(nn.Module):
    """
    Reconstructs the OLD-A reduction cell so that:
      (a) The forward pass is architecturally correct for non-standard input
          sizes (e.g. darcyflow 85x85) where the new AvgPool+Conv approach
          produces a spatial-size mismatch.
      (b) The state-dict keys exactly match the OLD-A .pth format:
            layers.0.preprocess.op.{1,2}  - 1x1 stride-2 channel projection
            layers.2.op.{1,2}             - first  3x3 conv
            layers.3.conv / .bn           - 1x1 stride-2 shortcut
            layers.5.op.{1,2}             - second 3x3 conv
          Positions 1 and 4 are nn.Identity (no params), so numbering
          skips 0->2->3->5 in the state dict.
    """
    def __init__(self, inplanes, planes, stride=2, affine=True):
        super().__init__()
        assert stride == 2
        self.layers = nn.ModuleList([
            _PreprocessWrapper(inplanes, planes, stride, affine),   # 0
            nn.Identity(),                                           # 1  no params
            ReLUConvBN(planes, planes, 3, 1, 1, 1, affine),        # 2
            _ConvBNShortcut(inplanes, planes, stride, affine),      # 3
            nn.Identity(),                                           # 4  no params
            ReLUConvBN(planes, planes, 3, 1, 1, 1, affine),        # 5
        ])
        self.in_dim  = inplanes
        self.out_dim = planes

    def forward(self, x):
        out = self.layers[0](x)    # 1x1 stride-2 projection
        out = self.layers[1](out)  # identity
        out = self.layers[2](out)  # first 3x3 conv
        res = self.layers[3](x)    # 1x1 stride-2 shortcut
        out = out + res
        out = self.layers[4](out)  # identity
        out = self.layers[5](out)  # second 3x3 conv
        return out


class LegacyTinyNetwork(nn.Module):
    """
    Mirrors TinyNetwork but (a) uses LegacyResNetBasicblock for reduction
    cells and (b) accepts an explicit in_channels parameter so it can handle
    both darcyflow (3ch) and ninapro (1ch) inputs.
    """
    def __init__(self, C, N, genotype, num_classes, in_channels=3):
        super().__init__()
        self._C      = C
        self._layerN = N

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )

        layer_channels   = [C]*N + [C*2] + [C*2]*N + [C*4] + [C*4]*N
        layer_reductions = [False]*N + [True] + [False]*N + [True] + [False]*N

        C_prev = C
        self.cells = nn.ModuleList()
        for C_curr, reduction in zip(layer_channels, layer_reductions):
            cell = (LegacyResNetBasicblock(C_prev, C_curr, 2, True)
                    if reduction
                    else InferCell(genotype, C_prev, C_curr, 1))
            self.cells.append(cell)
            C_prev = cell.out_dim

        self._Layer       = len(self.cells)
        self.lastact      = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier   = nn.Linear(C_prev, num_classes)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_format(state_dict):
    """
    Inspect reduction-cell keys and return:
      'new'    - conv_a / conv_b / downsample  (ninapro)
      'old-a'  - layers.0.preprocess / layers.2 / layers.3.conv+bn / layers.5
                 ONLY (exact match, no extra layers.1)
      'old'    - any other layers.N.* pattern
    """
    cell5_keys = {k.replace('cells.5.', '') for k in state_dict
                  if k.startswith('cells.5.')}

    if 'conv_a.op.1.weight' in cell5_keys:
        return 'new'

    old_a_required = {
        'layers.0.preprocess.op.1.weight',
        'layers.2.op.1.weight',
        'layers.3.conv.weight',
        'layers.5.op.1.weight',
    }
    extra_at_1 = {k for k in cell5_keys if k.startswith('layers.1.')}
    if old_a_required.issubset(cell5_keys) and not extra_at_1:
        return 'old-a'

    return 'old'


def _get_input_channels(state_dict):
    """Read the number of input channels from stem.0.weight."""
    return state_dict['stem.0.weight'].shape[1]


def _dummy_input(pth_data, dataset_key, state_dict):
    """
    Build a (2, C, H, W) random tensor matching the network's expected input.

    xshape format varies by dataset:
      darcyflow  (1, H, W, C)  NHWC -> C = xshape[3],  H/W = xshape[1:3]
      ninapro    (1, C, H, W)  NCHW -> C = xshape[1],  H/W = xshape[2:4]
    We resolve ambiguity by comparing with the actual stem input channels.
    """
    xshape  = pth_data[dataset_key]['opt_config']['xshape']
    C_in    = _get_input_channels(state_dict)
    if xshape[-1] == C_in:          # NHWC (darcyflow)
        H, W = xshape[1], xshape[2]
    else:                            # NCHW (ninapro)
        H, W = xshape[2], xshape[3]
    return torch.randn(2, C_in, H, W)


# ── Public API ────────────────────────────────────────────────────────────────

def load_pth_network(pth_path):
    """
    Load a NB201 network from a .pth file (DarcyFlow or NinaPro).

    Args:
        pth_path : path to an  arch-XXXXXX-seed-YYYY.pth  file

    Returns:
        network      - nn.Module in eval() mode
        dataset_key  - 'darcyflow' or 'ninapro'
        arch_config  - dict with C, N, genotype, num_classes
        fmt          - 'new', 'old-a', or 'old'
        missing_keys - list of model keys not loaded from file
    """
    data        = torch.load(pth_path, map_location='cpu')
    dataset_key = data['all_dataset_keys'][0]
    inner       = data[dataset_key]
    cfg         = inner['arch_config']
    sd          = inner['net_state_dict']

    fmt    = _detect_format(sd)
    C_in   = _get_input_channels(sd)

    if fmt == 'new':
        # Ninapro new-format: use standard get_cell_based_tiny_net, then patch
        # the stem to accept C_in (1) channels instead of the hardcoded 3.
        network = get_cell_based_tiny_net(cfg)
        network.stem[0] = nn.Conv2d(C_in, cfg['C'], kernel_size=3,
                                    padding=1, bias=False)
        network.load_state_dict(sd)
        missing = []

    elif fmt == 'old-a':
        # Exact OLD-A darcyflow layout: LegacyTinyNetwork loads 100% weights.
        network = LegacyTinyNetwork(cfg['C'], cfg['N'], cfg['genotype'],
                                    cfg['num_classes'], in_channels=C_in)
        network.load_state_dict(sd)
        missing = []

    else:
        # Other old darcyflow variant: legacy architecture is correct for
        # 85x85 inputs; load with strict=False.  Normal cells (15/17) load
        # correctly; only the 2 reduction cells stay at random init.
        network = LegacyTinyNetwork(cfg['C'], cfg['N'], cfg['genotype'],
                                    cfg['num_classes'], in_channels=C_in)
        result  = network.load_state_dict(sd, strict=False)
        missing = result.missing_keys

    network.eval()
    return network, dataset_key, cfg, fmt, missing


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    pth_dir   = 'all/all'
    pth_files = sorted(f for f in os.listdir(pth_dir) if f.endswith('.pth'))

    print(f'Testing load_pth_network on all {len(pth_files)} .pth files\n')

    counts = {'new': 0, 'old-a': 0, 'old': 0}
    for fname in pth_files:
        path = os.path.join(pth_dir, fname)
        net, ds, cfg, fmt, missing = load_pth_network(path)

        raw   = torch.load(path, map_location='cpu')
        dummy = _dummy_input(raw, ds, raw[ds]['net_state_dict'])
        counts[fmt] += 1

        with torch.no_grad():
            _, logits = net(dummy)

        n_total   = len(net.state_dict())
        n_missing = len(missing)
        pct       = 100 * (n_total - n_missing) / n_total
        status    = 'FULL (100%)' if not missing else f'PARTIAL ({pct:.0f}%)'

        print(f'  {fname}')
        print(f'    dataset / format  : {ds} / {fmt}')
        print(f'    num_classes       : {cfg["num_classes"]}')
        print(f'    input / logits    : {tuple(dummy.shape)} / {tuple(logits.shape)}')
        print(f'    weight load       : {status}')
        print()

    print('=== Summary ===')
    print(f'  new   (ninapro,    100% weights) : {counts["new"]:2d} files')
    print(f'  old-a (darcyflow,  100% weights) : {counts["old-a"]:2d} files')
    print(f'  old   (darcyflow, ~89% weights)  : {counts["old"]:2d} files')
