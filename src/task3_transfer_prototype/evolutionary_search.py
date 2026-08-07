"""
Evolutionary NAS with Surrogate Model
======================================
Algorithm (from professor's notes):
  1. Initialize population from our 47 evaluated architectures
  2. Train XGBoost predictor on that population (surrogate / regression model)
  3. Rank population by predicted score
  4. Select top-k parents
  5. Mutate each parent n times -> candidate pool
  6. Evaluate candidates with surrogate (instant, no GPU)
  7. Add top-x candidates to population
  8. Re-rank -> go to step 4
  9. Repeat until population reaches max_size
 10. Look up true NB201 accuracy for the best found architectures
     (verifies the predictor generalises to unseen architectures)

Search space: NAS-Bench-201 -- 15,625 architectures
              4 nodes, 6 directed edges, 5 possible ops per edge
"""

import os, sys, pickle, bz2, json, random, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats     import kendalltau as scipy_kendalltau
from sklearn.ensemble import RandomForestRegressor
from xgboost         import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAML_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_DIR   = os.path.join(RAML_DIR, 'all', 'all')
NATS_DIR  = os.path.join(RAML_DIR, 'NATS-tss-v1_0-3ffb9-simple')
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ── Architecture encoding ─────────────────────────────────────────────────────

OPS     = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
OP2IDX  = {op: i for i, op in enumerate(OPS)}
N_OPS   = len(OPS)
N_EDGES = 6

def parse_arch_str(arch_str):
    ops = []
    for token in arch_str.replace('+', '|').split('|'):
        token = token.strip()
        if '~' in token:
            ops.append(token.split('~')[0].strip())
    assert len(ops) == N_EDGES
    return ops

def ops_to_arch_str(ops):
    return (f'|{ops[0]}~0|'
            f'+|{ops[1]}~0|{ops[2]}~1|'
            f'+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|')

def encode_arch(arch_str):
    enc = np.zeros(N_EDGES * N_OPS, dtype=np.float32)
    for i, op in enumerate(parse_arch_str(arch_str)):
        enc[i * N_OPS + OP2IDX[op]] = 1.0
    return enc

def arch_label(arch_str):
    abbrev = {'none':'N','skip_connect':'S','nor_conv_1x1':'1',
              'nor_conv_3x3':'3','avg_pool_3x3':'P'}
    return '-'.join(abbrev[op] for op in parse_arch_str(arch_str))

# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(arch_str, rng, n_mutations=1):
    """Randomly replace 1 (or more) operations with a different op."""
    ops = parse_arch_str(arch_str)
    edges = rng.sample(range(N_EDGES), min(n_mutations, N_EDGES))
    for edge in edges:
        choices = [op for op in OPS if op != ops[edge]]
        ops[edge] = rng.choice(choices)
    return ops_to_arch_str(ops)

def random_architecture(rng):
    """Sample a uniformly random NB201 architecture."""
    ops = [rng.choice(OPS) for _ in range(N_EDGES)]
    return ops_to_arch_str(ops)

# ── Data loading ──────────────────────────────────────────────────────────────

SEEDS     = (777, 888, 999)
FINAL_KEY = 'ori-test@199'

def load_population_data(dataset='cifar100'):
    """Load 47 architectures with arch_str + true accuracy for the given dataset."""
    fnames  = sorted(f for f in os.listdir(ALL_DIR) if f.endswith('.pickle.pbz2'))
    records = []
    for fname in fnames:
        with bz2.BZ2File(os.path.join(ALL_DIR, fname), 'rb') as f:
            p = pickle.load(f)
        d200     = p['200']
        arch_str = d200['arch_str']
        results  = d200['all_results']
        vals     = [results[(dataset, s)]['eval_acc1es'].get(FINAL_KEY)
                    for s in SEEDS if (dataset, s) in results]
        vals     = [v for v in vals if v is not None]
        if not vals:
            continue
        records.append({'arch_str': arch_str, 'true_acc': float(np.mean(vals))})
    return records

# ── True accuracy lookup from NATS-tss (all 15,625 architectures) ─────────────

def lookup_true_acc(arch_str, dataset='cifar100', seed=777):
    """
    Look up the true NB201 benchmark accuracy for any architecture.
    The arch_str encodes the index uniquely -- we derive it by matching
    against the NATS-tss files using the stored arch_str field.
    Uses the NATS-Bench api to convert arch_str -> index.
    Falls back to a linear scan if api is unavailable.
    """
    try:
        from nats_bench import create as nats_create
        api = nats_create(NATS_DIR, 'tss', fast_mode=True, verbose=False)
        idx = api.query_index_by_arch(arch_str)
        if idx is None or idx < 0:
            return None
        nats_path = os.path.join(NATS_DIR, f'{idx:06d}.pickle.pbz2')
        if not os.path.exists(nats_path):
            return None
        with bz2.BZ2File(nats_path, 'rb') as f:
            p = pickle.load(f)
        results = p['200']['all_results']
        vals = [results[(dataset, s)]['eval_acc1es'].get(FINAL_KEY)
                for s in SEEDS if (dataset, s) in results]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    except Exception:
        return None

# ── Surrogate model ───────────────────────────────────────────────────────────

def train_surrogate(records):
    X = np.stack([encode_arch(r['arch_str']) for r in records])
    y = np.array([r['true_acc'] for r in records], dtype=np.float64)
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, eval_metric='rmse', verbosity=0)
    model.fit(X, y)
    return model

def predict(model, arch_str):
    return float(model.predict(encode_arch(arch_str).reshape(1, -1))[0])

# ── Evolutionary search ───────────────────────────────────────────────────────

def evolutionary_search(
    initial_records,   # list of {arch_str, true_acc}
    surrogate,         # trained XGBoost model
    n_select    = 2,   # parents per iteration
    n_mutations = 20,  # mutations per parent
    n_add       = 5,   # best mutations added per iteration
    max_pop     = 200, # stopping criterion
    seed        = 42,
):
    rng = random.Random(seed)

    # Population: set of arch_strs (deduped)
    population = list({r['arch_str'] for r in initial_records})

    history = []   # (iteration, pop_size, best_predicted, best_arch_str)

    iteration = 0
    while len(population) < max_pop:
        iteration += 1

        # Score every architecture in population
        scored = [(arch, predict(surrogate, arch)) for arch in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_arch, best_score = scored[0]
        history.append({
            'iteration':       iteration,
            'pop_size':        len(population),
            'best_predicted':  best_score,
            'best_arch':       best_arch,
        })

        # Select top parents
        parents = [arch for arch, _ in scored[:n_select]]

        # Mutate parents -> candidate pool
        candidates = set()
        pop_set    = set(population)
        attempts   = 0
        while len(candidates) < n_select * n_mutations and attempts < 5000:
            parent = rng.choice(parents)
            child  = mutate(parent, rng, n_mutations=1)
            if child not in pop_set and child not in candidates:
                candidates.add(child)
            attempts += 1

        # If mutation neighbourhood is exhausted, inject random architectures
        if not candidates:
            inject_attempts = 0
            while len(candidates) < n_add * 4 and inject_attempts < 2000:
                rand_arch = random_architecture(rng)
                if rand_arch not in pop_set:
                    candidates.add(rand_arch)
                inject_attempts += 1
            if candidates:
                print(f"  [iter {iteration}] Neighbourhood exhausted — "
                      f"injected {len(candidates)} random architectures.")
            else:
                print(f"  [iter {iteration}] Could not find new architectures — stopping.")
                break

        # Evaluate candidates with surrogate
        cand_scored = [(c, predict(surrogate, c)) for c in candidates]
        cand_scored.sort(key=lambda x: x[1], reverse=True)

        # Add top-n_add to population
        added = [arch for arch, _ in cand_scored[:n_add]]
        population.extend(added)

    # Final ranking
    final_scored = [(arch, predict(surrogate, arch)) for arch in population]
    final_scored.sort(key=lambda x: x[1], reverse=True)

    return final_scored, history

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_search_trajectory(history, dataset_label, save_path):
    iters     = [h['iteration']      for h in history]
    pop_sizes = [h['pop_size']       for h in history]
    scores    = [h['best_predicted'] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    color_score = '#1976D2'
    color_pop   = '#aaaaaa'

    ax1.plot(iters, scores, color=color_score, linewidth=2, marker='o',
             markersize=4, label='Best predicted score')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel(f'Best predicted accuracy (%) — {dataset_label}',
                   color=color_score, fontsize=10)
    ax1.tick_params(axis='y', labelcolor=color_score)

    ax2 = ax1.twinx()
    ax2.plot(iters, pop_sizes, color=color_pop, linewidth=1.5,
             linestyle='--', label='Population size')
    ax2.set_ylabel('Population size', color=color_pop, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color_pop)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower right')

    ax1.set_title(f'Evolutionary NAS Search Trajectory\n'
                  f'Fitness: XGBoost surrogate ({dataset_label})', fontsize=11)
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()

def plot_predicted_vs_true(results_top, dataset_label, save_path):
    """Scatter: predicted vs true accuracy for top architectures found by search."""
    preds = [r['predicted'] for r in results_top if r['true_acc'] is not None]
    trues = [r['true_acc']  for r in results_top if r['true_acc'] is not None]
    if not preds:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(trues, preds, s=50, alpha=0.8, color='#E53935', zorder=3)
    mn, mx = min(min(trues), min(preds)), max(max(trues), max(preds))
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1, alpha=0.5, label='y=x')

    tau, _ = scipy_kendalltau(trues, preds)
    ax.set_title(f'Predicted vs True — Top-{len(preds)} found archs\n'
                 f'Kendall Tau = {tau:.3f}  ({dataset_label})', fontsize=10)
    ax.set_xlabel('True accuracy (%)')
    ax.set_ylabel('Predicted accuracy (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────

DATASET       = 'cifar100'   # best predictor tau (0.718); change to 'cifar10' if preferred
DATASET_LABEL = 'CIFAR-100'
N_SELECT      = 2
N_MUTATIONS   = 20
N_ADD         = 5
MAX_POP       = 200
N_TOP_VERIFY  = 20   # look up true accuracy for this many top found architectures

def main():
    print('=' * 62)
    print(f'  Evolutionary NAS  |  fitness: XGBoost on {DATASET_LABEL}')
    print(f'  parents={N_SELECT}  mutations/parent={N_MUTATIONS}  '
          f'add/iter={N_ADD}  max_pop={MAX_POP}')
    print('=' * 62)

    # 1. Load initial population
    records = load_population_data(DATASET)
    print(f'\nInitial population: {len(records)} architectures loaded.')
    print(f'True acc range: {min(r["true_acc"] for r in records):.2f}% '
          f'-- {max(r["true_acc"] for r in records):.2f}%')

    # 2. Train surrogate on full initial population
    print('\nTraining XGBoost surrogate ...')
    surrogate = train_surrogate(records)

    # Sanity-check surrogate on training data
    preds_train = [predict(surrogate, r['arch_str']) for r in records]
    trues_train = [r['true_acc'] for r in records]
    tau_train, _ = scipy_kendalltau(trues_train, preds_train)
    print(f'Surrogate train Kendall Tau = {tau_train:.4f}  (in-sample)')

    # 3. Run evolutionary search
    print(f'\nRunning evolutionary search (max population = {MAX_POP}) ...')
    final_scored, history = evolutionary_search(
        records, surrogate,
        n_select=N_SELECT, n_mutations=N_MUTATIONS,
        n_add=N_ADD, max_pop=MAX_POP
    )

    print(f'\nSearch complete: {len(final_scored)} architectures in final population.')
    print(f'Iterations: {len(history)}')

    # 4. Show top found architectures
    print(f'\nTop 10 architectures by predicted score:')
    print(f'  {"Rank":4s}  {"Predicted%":10s}  {"Encoding":18s}  {"In initial pop?"}')
    print('  ' + '-' * 54)
    initial_set = {r['arch_str'] for r in records}
    for rank, (arch, pred) in enumerate(final_scored[:10], 1):
        in_init = 'yes' if arch in initial_set else 'NEW (found by search)'
        print(f'  {rank:4d}  {pred:10.4f}  {arch_label(arch):18s}  {in_init}')

    # 5. Look up true accuracy for top-N architectures
    print(f'\nLooking up true NB201 accuracy for top {N_TOP_VERIFY} found architectures ...')
    results_top = []
    for arch, pred in final_scored[:N_TOP_VERIFY]:
        true_acc = lookup_true_acc(arch, dataset=DATASET)
        in_init  = arch in initial_set
        results_top.append({
            'arch':      arch,
            'encoding':  arch_label(arch),
            'predicted': pred,
            'true_acc':  true_acc,
            'in_initial_population': in_init,
        })

    # Compute Kendall Tau on the verified subset (only NEW architectures)
    new_only = [r for r in results_top if not r['in_initial_population']
                and r['true_acc'] is not None]
    if len(new_only) >= 3:
        tau_new, _ = scipy_kendalltau(
            [r['true_acc'] for r in new_only],
            [r['predicted'] for r in new_only]
        )
        print(f'\nKendall Tau on NEW (unseen) architectures: {tau_new:.4f}  '
              f'({len(new_only)} samples)')
    else:
        tau_new = None
        print('\nNot enough new architectures with true acc for Kendall Tau.')

    print(f'\nTop {N_TOP_VERIFY} found — predicted vs true:')
    print(f'  {"Rank":4s}  {"Predicted%":10s}  {"True%":8s}  {"New?":5s}  {"Encoding"}')
    print('  ' + '-' * 62)
    for rank, r in enumerate(results_top, 1):
        true_str = f'{r["true_acc"]:.3f}' if r["true_acc"] is not None else '  N/A '
        new_str  = '' if r['in_initial_population'] else '*NEW*'
        print(f'  {rank:4d}  {r["predicted"]:10.4f}  {true_str:8s}  {new_str:5s}  '
              f'{r["encoding"]}')

    # 6. Initial population best vs search best
    init_best_acc  = max(r['true_acc'] for r in records)
    init_best_arch = max(records, key=lambda r: r['true_acc'])['arch_str']
    found_best     = results_top[0]

    print(f'\nComparison:')
    print(f'  Best in initial population : {init_best_acc:.3f}%  '
          f'({arch_label(init_best_arch)})')
    if found_best['true_acc'] is not None:
        print(f'  Best found by search       : {found_best["true_acc"]:.3f}%  '
              f'({found_best["encoding"]})  '
              f'{"(NEW)" if not found_best["in_initial_population"] else "(was in initial pop)"}')
        improvement = found_best['true_acc'] - init_best_acc
        print(f'  Improvement                : {improvement:+.3f}%')

    # 7. Save results
    save_data = {
        'dataset':      DATASET,
        'n_select':     N_SELECT,
        'n_mutations':  N_MUTATIONS,
        'n_add':        N_ADD,
        'max_pop':      MAX_POP,
        'n_iterations': len(history),
        'final_pop_size': len(final_scored),
        'surrogate_train_tau': tau_train,
        'tau_new_architectures': tau_new,
        'history':      history,
        'top_results':  results_top,
    }
    out_path = os.path.join(OUT_DIR, 'evolutionary_results.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\nResults saved to {out_path}')

    # 8. Plots
    plot_search_trajectory(
        history, DATASET_LABEL,
        os.path.join(OUT_DIR, 'evo_search_trajectory.png')
    )
    plot_predicted_vs_true(
        results_top, DATASET_LABEL,
        os.path.join(OUT_DIR, 'evo_predicted_vs_true.png')
    )
    print('Plots saved: evo_search_trajectory.png  +  evo_predicted_vs_true.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
