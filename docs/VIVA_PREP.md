# RAML Viva Preparation — The Complete Study Guide
### Everything we did in Tasks 1–5, every concept behind it, explained from zero

**Read this top to bottom once, then use Part F (cheat sheet) and Part G (Q&A) for revision.**

---

# PART A — The Big Picture (start here)

## A.1 The one-paragraph story of the whole project

Scientists have a telescope problem: images from the Hubble Space Telescope are
polluted by **cosmic rays** — high-energy particles that leave bright streaks on the
camera sensor. We want a neural network that looks at a telescope image and marks,
pixel by pixel, "this pixel is a cosmic ray, this one is not." But *which* neural
network design (architecture) should we use? There are thousands of possible designs.
Training every design to find the best one would take months of GPU time. So the
project builds a **shortcut**: a cheap "fortune teller" model (a *performance
predictor*) that can look at a network's blueprint and predict how good it would be
**without training it**. We use that predictor to pick the most promising designs,
then take those few designs, reuse knowledge they learned on ordinary image datasets
(**transfer learning**), and fine-tune them for the cosmic-ray task.

That's it. Everything in Tasks 1–5 serves that story:

| Task | Role in the story |
|------|-------------------|
| 1 | Learn to *load* pre-trained networks from files (the plumbing) |
| 2 | *Rank* the given pre-trained networks — which is best per dataset? |
| 3 | First *transfer learning*: fine-tune one network onto Cosmic + first small predictor + evolutionary search |
| 4 | Scale up: predictor on **all 15,625** architectures + random search + fine-tune 30 networks |
| 5 | Do task 4 *properly to the professor's targets*: better predictor (all 9 Kendall-Tau targets met), better selection, and real pre-trained weights (via the Omni cluster) |

## A.2 The three-line elevator answers

If the professor asks **"what did you do?"**:
> "We built a performance predictor that ranks all 15,625 NAS-Bench-201 architectures
> from their blueprints alone, hitting all nine Kendall-Tau targets. We then used it
> inside a random search to select the 10 most promising networks per dataset, and
> fine-tuned those 30 networks on the Cosmic ray segmentation task, reaching
> 0.94–0.98 AUROC."

If asked **"what was the hardest part?"**:
> "Two things: making the predictor work with only 32 training examples — solved by
> switching from tree ensembles to a rank-averaged SVR ensemble — and obtaining the
> official pre-trained weights, which only exist inside a 360 GB archive; we moved
> that download to the Omni cluster and extract only the 18 networks we need."

---

# PART B — Machine Learning from Zero

## B.1 What is a model?

A **model** is a mathematical function with adjustable knobs. Input goes in (an
image), output comes out (a prediction). The knobs are called **parameters** or
**weights** — millions of numbers. "Training" means automatically turning those
knobs until the outputs are right.

**Analogy:** a giant mixing desk in a music studio with a million sliders. Training
is an automatic process that nudges sliders until the music sounds right.

## B.2 How does training work?

1. **Loss function** — a number measuring "how wrong" the model currently is.
   Lower = better. (Ours: binary cross-entropy, see E.7.)
2. **Gradient descent** — calculus tells us, for every weight, which direction to
   nudge it to reduce the loss. We nudge all weights a tiny step. Repeat thousands
   of times.
3. **Learning rate (lr)** — the step size of each nudge. Too big: you overshoot and
   destroy things. Too small: training takes forever. (We used 1e-3 for the head,
   1e-4 for the full network — deliberately smaller to be gentle, see E.9.)
4. **Epoch** — one full pass through the training data. We used 3 + 5 = 8 epochs.
5. **Batch** — we feed images in groups (batch size 32) instead of one at a time,
   for speed and stability.

## B.3 Train / validation / test — why three datasets?

- **Training set** — the model learns from this.
- **Validation set** — used to *choose* things (best epoch, best settings). The
  model never trains on it.
- **Test set** — touched only at the very end, to report honest final numbers.

**Why?** A student who memorises the exercise sheet aces the exercises but fails the
exam. Checking on unseen data catches "memorising" (called **overfitting**).

## B.4 Overfitting — the single most important concept in our project

**Overfitting** = the model learns the training examples *by heart*, including their
noise, instead of learning the general pattern. Symptom: great on training data,
bad on new data. It happens most when you have **few training examples and a very
flexible model**.

This is *the* reason Task 5's predictor switched from XGBoost to SVR: with only 32
training architectures, a 500-tree XGBoost memorised them (rank quality ≈ 0.30);
the smoother SVR generalised (≈ 0.44). Remember this — it's the best answer to
"why SVR?".

## B.5 What is a neural network / CNN?

A **neural network** is layers of simple units: each takes numbers in, multiplies
by weights, adds up, applies a simple non-linearity (e.g. **ReLU**, which just
zeroes out negatives), passes on.

A **CNN (convolutional neural network)** is a neural net specialised for images:
- A **convolution** slides a small window (e.g. 3×3 pixels) across the image,
  computing the same little pattern-detector at every position. Early layers detect
  edges; deeper layers detect textures, shapes, objects.
- **1×1 convolution** — looks at a single pixel position across all channels; used
  to mix/reduce channels cheaply.
- **Pooling** (e.g. 3×3 average pooling) — shrinks the image by averaging
  neighbourhoods; makes the network see "bigger picture".
- **Skip connection** — a wire that copies the input straight to the output,
  bypassing the layer. Helps very deep networks train (this idea is ResNet).
- **Batch normalisation (BN)** — keeps the numbers flowing through the net in a
  healthy range; stabilises training.

## B.6 Classification vs segmentation

- **Classification:** one label for the whole image ("this is a cat"). CIFAR-10 etc.
- **Segmentation:** one label **per pixel** ("these pixels are cosmic ray, those are
  background"). The Cosmic task is **binary segmentation** — output is a 128×128
  mask of yes/no per pixel.

---

# PART C — The Building Blocks of Our Project

## C.1 What is NAS (Neural Architecture Search)?

Designing a network architecture (how many layers, which operations, how connected)
is normally done by human experts. **NAS = letting an algorithm search for the
architecture automatically.** Every NAS method has three ingredients:

1. **Search space** — the set of allowed designs (ours: NAS-Bench-201's 15,625).
2. **Search strategy** — how to explore it (ours: random sampling + predictor
   ranking; task 3 also tried evolutionary search).
3. **Performance estimation** — how to judge a candidate without full training
   (ours: the performance predictor — the heart of this project).

## C.2 NAS-Bench-201 — our search space (memorise this section!)

NAS-Bench-201 (NB201) is a **public benchmark**: researchers pre-trained **all
15,625 architectures** of a small search space on three datasets and published every
result. It's a giant lookup table — you can "evaluate" an architecture instantly by
looking up its accuracy, no GPU needed. That's what makes predictor research
possible: we have ground-truth accuracy for every single architecture.

**The search space:** every architecture is one **cell**, repeated many times. The
cell is a tiny directed graph (**DAG**) with 4 nodes (node 0 = input, node 3 =
output) and **6 edges**. On each edge you place one of **5 operations**:

| Code | Operation | Meaning |
|------|-----------|---------|
| N | `none` | the edge is cut (no connection) |
| S | `skip_connect` | pass the input through unchanged |
| 1 | `nor_conv_1x1` | 1×1 convolution |
| 3 | `nor_conv_3x3` | 3×3 convolution |
| P | `avg_pool_3x3` | 3×3 average pooling |

6 edges × 5 choices each = **5⁶ = 15,625 architectures**. Each is written as an
**architecture string** like:
`|nor_conv_3x3~0|+|none~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|none~2|`
(read: what op sits on each edge, `~k` = coming from node k). We abbreviate that
one as `3-N-P-S-3-N`.

**Edge numbering (needed to explain path encoding):**
e0: 0→1, e1: 0→2, e2: 1→2, e3: 0→3, e4: 1→3, e5: 2→3.

**The macro skeleton (TinyNetwork):** the cell alone isn't a network. NB201 wraps
it: a stem (3×3 conv), then 3 stages of 5 identical cells, with a fixed
downsampling **ResNet basic block** between stages, then pooling + classifier.
Only the cell varies between the 15,625 — the skeleton is fixed.

**The three datasets networks were trained on:**
- **CIFAR-10** — 32×32 colour photos, 10 classes (plane, car, bird, …)
- **CIFAR-100** — same images, 100 fine-grained classes (harder)
- **ImageNet16-120** — mini-ImageNet: 16×16 pixels, 120 classes (hardest)

## C.3 NAS-Bench-360 and the "downstream tasks"

NAS-Bench-360 asks: do architectures found on standard datasets work on *diverse
real-world tasks*? Its tasks include:
- **DarcyFlow** — predict fluid flow through porous material (physics/PDE), 85×85 inputs.
- **NinaPro** — classify hand gestures from EMG muscle signals, 16×52 single-channel inputs.
- **Cosmic** — our target: detect cosmic-ray pixels in Hubble images.

Tasks 1–2 worked with pre-trained DarcyFlow/NinaPro networks the professor provided.

## C.4 The Cosmic dataset (our target task)

- Source: **deepCR ACS-WFC** — real Hubble Space Telescope images.
- We use filter **f435w**, **GAL** (galaxy) fields only — matching the deepCR paper.
- **4,347 train / 483 validation / 420 test** patches of **128×128** pixels,
  single channel (grayscale astronomy image).
- Label: a 128×128 **binary mask** — 1 = cosmic ray pixel, 0 = clean. Some pixels
  are flagged **-9999 (IGNORE)** — near bright stars/chip edges where nobody can
  say — and are excluded from loss and metric.

---

# PART D — Task-by-Task Walkthrough

## D.1 Task 1 — Loading the pre-trained weight files (the plumbing)

**What we were given:** 20 `.pth` files (PyTorch checkpoint files) — 10 DarcyFlow +
10 NinaPro networks, trained by others, possibly with *older* library versions.

**What "loading weights" means:** a checkpoint is a **state_dict** — a dictionary
mapping layer names → weight tensors, e.g. `"stem.0.weight" → [16×3×3×3 numbers]`.
To use it you must build the *exact same* architecture in code, then copy the
numbers in. If names or shapes don't match, it fails.

**The three real problems we solved:**

1. **Spatial size mismatch (DarcyFlow, 85×85 inputs).** The modern library's
   downsampling block uses average-pooling (stride 2) on the shortcut path:
   85→42 pixels. The conv path gives 85→43. 42 ≠ 43 → crash when adding the two
   paths. The *old* library used matched 1×1 stride-2 convs on both paths.
   **Fix:** we re-implemented the legacy block (`LegacyResNetBasicblock`).
2. **Layer-name (key) format mismatch.** Old checkpoints name reduction-cell
   weights `layers.N.*`; new code expects `conv_a/conv_b/downsample`. DarcyFlow
   files are old-format, NinaPro new-format. **Fix:** detect format per file and
   build/rename accordingly.
3. **Input channels.** The library hardcodes a 3-channel (RGB) stem; NinaPro data
   is 1-channel. **Fix:** patch the stem layer to 1 channel after construction,
   before loading weights.

**Lesson to state in viva:** "reproducing research code is largely about
version-drift archaeology — the maths was fine, the engineering was the work."

## D.2 Task 2 — Ranking the given networks (which architecture is best where?)

**Goal:** among the provided files, find the best architecture per dataset.

**What we did:** scanned 27 `.pickle.pbz2` result files (CIFAR-10 / CIFAR-100 /
ImageNet16-120) and the 20 `.pth` files (DarcyFlow / NinaPro), read each network's
**final-epoch (epoch 199)** evaluation metric, averaged over training seeds, and
printed ranked tables.

**Detail that shows care:** metrics differ per dataset — for DarcyFlow lower
validation *loss* is better; for NinaPro higher validation *accuracy* is better; for
CIFAR/ImageNet it's test accuracy. You must sort in the right direction!

**Key output:** the best NinaPro architecture (**arch 006186**, score 94.39%)
became the backbone for Task 3's transfer learning.

## D.3 Task 3 — First transfer learning onto Cosmic (+ first predictor & evolutionary search)

### D.3.1 The transfer-learning experiment

**Idea:** take the best NinaPro network (it already "knows" how to extract features
from 1-channel signal images), replace its classification head with a segmentation
head, and fine-tune it on Cosmic.

**Why NinaPro as source?** (a) NinaPro nets have a 1-channel stem — directly
compatible with grayscale Cosmic images; (b) its weights loaded 100% cleanly;
(c) it was the best performer on its own task.

**The architecture we built (371,473 parameters):**
```
input (1×128×128)
 → NB201 TinyNetwork backbone (stem → 5 cells → downsample → 5 cells → downsample → 5 cells → BN+ReLU)
 → feature map (64 channels × 32×32)
 → EinspaceCosmicHead:
      AdaptiveAvgPool2d(128×128)   (resize feature map back to image size)
      Conv2d(64→1, 1×1)            (one score per pixel)
 → output logit map (1×128×128)
```

**Two-phase fine-tuning (memorise the rationale):**

| Phase | What trains | LR | Why |
|-------|-------------|-----|-----|
| 1 (3 epochs) | Head only — backbone frozen | 1e-3 | The new head is random; let it adapt first while the pre-trained backbone stays intact |
| 2 (5 epochs) | Everything | 1e-4 | Now let the backbone specialise, but gently — a big LR would destroy ("catastrophically forget") its pre-trained features |

**Results (know these numbers):**

| Stage | Val AUROC | Test AUROC |
|-------|-----------|------------|
| Baseline (random head, no training) | 0.489 | 0.444 |
| Phase 1 (head only) | 0.589 | 0.601 |
| Phase 2 (full fine-tune) | **0.976** | **0.944** |

Interpretation: baseline ≈ 0.5 = coin flip, as expected. Phase 1 alone is weak
because the head has only **65 trainable parameters** (64 weights + 1 bias) — it
can't fix features that were learned for muscle signals. Phase 2 jumps to 0.95 in
a single epoch — once the backbone can adapt, it specialises fast. Final test 0.944
is close to the specialised deepCR reference model (~0.97) with only 8 epochs.

### D.3.2 The first (small) performance predictor

With only **47 architectures** that had result files, we trained Random Forest and
XGBoost to map a 30-dim architecture encoding → accuracy, and measured **Kendall
Tau** for different training sizes p (repeated 50× to average noise). This was the
prototype for Tasks 4–5.

### D.3.3 Evolutionary search with a surrogate

Classic NAS strategy we implemented (professor's algorithm):
population of architectures → train XGBoost predictor on them (**the surrogate**) →
rank → keep top-k parents → **mutate** each (randomly change one edge's operation)
→ score children *with the predictor, no training* → add best children to
population → repeat. Finally, look up the true NB201 accuracy of the best found
architectures to check the predictor didn't fool us.

**Concept:** *evolution = selection + mutation; the surrogate makes fitness
evaluation free.*

## D.4 Task 4 — Scaling up: the full pipeline on all 15,625 architectures

The professor's email (this is "the last email task") asked for three things:
1. Improve the predictor using **all 15,625** architectures, with target Kendall-Tau
   values at p = 32 / 128 / 1024.
2. Randomly sample **300** networks, predict their accuracy, fine-tune the **best 10
   per dataset** (30 total) on Cosmic.
3. Download the NB201 **pre-trained weights** for fine-tuning.

**What task 4 did:**
- **Step 1:** built the accuracy cache for all 15,625 archs from the NATS archive
  (final-epoch test accuracy, mean over 3 seeds), encoded each arch as **30-dim
  one-hot**, trained XGBoost/RF, ran the p-sweep. Result: decent at p=1024 (~0.7)
  but **below the professor's targets, especially at small p** — this is what
  task 5 fixes.
- **Step 2:** random search: sample 300, rank by predictor trained on 1,024 archs,
  take top-10 per dataset. Kendall-Tau on the 300: **0.72–0.76**.
- **Step 3:** fine-tuned all 30 selected networks on Cosmic with the task-3 recipe
  (two-phase). All 30 succeeded: mean AUROC **0.958 / 0.953 / 0.959** (best single
  0.973). BUT — from **random initialisation**, because the weight files we had
  contain no weights (see D.5.4).

## D.5 Task 5 — Doing it properly: hitting all 9 targets

### D.5.1 The professor's targets

| Dataset | p=32 | p=128 | p=1024 |
|---------|------|-------|--------|
| CIFAR-10 | 0.39 | 0.54 | 0.74 |
| CIFAR-100 | 0.39 | 0.56 | 0.74 |
| ImageNet16-120 | 0.40 | 0.56 | 0.74 |

(p = how many labelled architectures the predictor may train on. Small p = "how
well can you rank with almost no data?" — the hard, practically-relevant regime.)

### D.5.2 Improvement 1 — ground truth from NASLib

The professor said "ground truth accessible from NASLib". We switched to reading
the **validation accuracy** of every architecture directly from NASLib's
`nb201_all.pickle` (CIFAR-10 from its `cifar10-valid` split). We verified
NASLib-validation vs NATS-test ground truth give near-identical Kendall-Tau — so
this alone wasn't the bottleneck, but it is exactly what was asked.

### D.5.3 Improvement 2 — path encoding (210-dim features)

The 30-dim one-hot says *which op sits on each edge*, independently. It cannot
express "edge A **and** edge B together form a conv chain". The **path encoding**
fixes that: the cell DAG has exactly **4 paths** from input to output:

| Path | Edges | Combinations |
|------|-------|--------------|
| 0→3 | e3 | 5¹ = 5 |
| 0→1→3 | e0,e4 | 5² = 25 |
| 0→2→3 | e1,e5 | 5² = 25 |
| 0→1→2→3 | e0,e2,e5 | 5³ = 125 |
| | total | **180** |

For each path, one-hot over the *combination of ops along it* — e.g. path 0→1→3
with (3×3 conv, 3×3 conv) lights up one specific feature of the 25. Final encoding
= 30 + 180 = **210 numbers** per architecture. This captures **operation
interactions**, which matter: information flows along paths, not edges in isolation.

### D.5.4 Improvement 3 — the right model: SVR rank-ensemble (THE key story)

With path features + XGBoost we reached only **5/9** targets. Diagnosis: at p=32 a
deep tree ensemble **overfits** (memorises 32 points; Kendall ≈ 0.30). We ran a
systematic model bake-off — Ridge, Bayesian Ridge, Gaussian Process, **SVR**,
Random Forest, gradient boosting — and SVR won decisively (see E.5 for how SVR
works). Final predictor:

> **Rank-average ensemble of two RBF-SVRs** — one tuned for small budgets
> (C=20, γ=0.02), one for large budgets (C=10, γ='scale'). Each model predicts,
> we convert each model's predictions to **ranks**, and average the ranks.

Why rank-averaging? Kendall-Tau only cares about *ordering*. Averaging ranks (not
raw scores) is invariant to each model's scale and directly targets the ordering.
Two extra facts we can defend: feature scaling (StandardScaler) *hurt* SVR here
(we tested); γ='scale' wins at large p while γ=0.02 wins at small p — the ensemble
gets both regimes.

**Final results — all 9 targets met (fixed 2,000-arch held-out test set, mean of
20 random training draws):**

| Dataset | p=32 (tgt) | p=128 (tgt) | p=1024 (tgt) |
|---------|-----------|-------------|--------------|
| CIFAR-10 | **0.439** (0.39) | **0.596** (0.54) | **0.741** (0.74) |
| CIFAR-100 | **0.437** (0.39) | **0.601** (0.56) | **0.750** (0.74) |
| ImageNet16-120 | **0.433** (0.40) | **0.596** (0.56) | **0.743** (0.74) |

### D.5.5 Selection re-run

Sampled 300 architectures (fixed seed 99 → reproducible), ranked with the ensemble
trained on p=1024, top-10 per dataset → **30 runs, 18 unique architectures**
(some are top-10 for multiple datasets — expected: good architectures are good
across datasets). Ranking quality on the 300: **0.758 / 0.776 / 0.760**.

### D.5.6 The pre-trained weights saga (great viva story — tell it confidently)

The professor said "download the weights from NAS-bench-201". Reality:
- The commonly-downloaded benchmark files (2.2 GB / 4.7 GB `.pth`) contain
  **metrics only — no weights** (the docs say so explicitly).
- The original 226 GB weights archive: **Google Drive link is dead (404)**.
- The current source (NATS-Bench "full"): **~360 GB in 12 split parts**; Google
  Drive **rate-limits anonymous downloads**; and it exceeded our local disk.
- We need only **18 of 15,625** networks — but a split tar archive cannot be
  cherry-picked; you must download all parts and reassemble before extracting.

**Solution:** delegate to the university's **Omni HPC cluster** (bandwidth, storage,
GPUs). We prepared a handoff kit (download script with authenticated-rclone
fallback, extraction script that pulls only the 18 files, a script that exports 30
small per-(architecture, dataset) weight files) plus a self-contained code bundle.
The definitive fine-tuning with real pre-trained weights runs there; meanwhile we
validated the whole pipeline end-to-end with random initialisation:
**AUROC 0.945–0.982 across all 30 networks** (ImageNet16-120-sourced networks best:
mean 0.968, best single 0.982 for arch `3-N-P-S-3-N`).

---

# PART E — Deep Dives on the Concepts You'll Be Grilled On

## E.1 Kendall Tau (know this cold)

**What:** a number in [-1, +1] measuring how similar two *rankings* are.
+1 = identical order, 0 = unrelated, -1 = exactly reversed.

**How it's computed:** take every **pair** of architectures (i, j). If the
predictor orders them the same way as the truth (both say i > j, or both say
i < j), the pair is **concordant**; otherwise **discordant**.

> **Tau = (concordant − discordant) / (total pairs)**

**Example:** truth ranks A > B > C. Predictor says A > C > B. Pairs: (A,B) ✓,
(A,C) ✓, (B,C) ✗ → tau = (2−1)/3 = **0.33**.

**Why rank correlation and not accuracy-error (e.g. MSE)?** Because we use the
predictor only to *pick the best* architectures. If it ranks perfectly but
predicts every accuracy 5 points too low, it's still perfect *for our purpose*.
Ranking is the right yardstick — and it's also why our rank-averaging ensemble is
principled: it optimises exactly what's measured.

**Intuition for values:** 0.74 tau is a strong ranking; 0.39 from 32 examples means
"even with 32 examples we're far better than chance."

## E.2 What is a performance predictor / surrogate model?

A small, cheap ML model that maps **architecture blueprint → predicted accuracy**.
Called a *surrogate* because it stands in for the expensive real evaluation
(training the network). Train it on p architectures whose accuracy you know; it
then scores all others instantly. Evaluation protocol we used: hold out a fixed
2,000-architecture test set, train on p random architectures from the rest, compute
tau on the test set, repeat 20× with different random draws, report the mean
(so results aren't luck of one draw).

## E.3 Encodings — why a network must become a vector

ML models eat numbers, not blueprints. An **encoding** turns the architecture into
a fixed-length numeric vector. **One-hot** means: for a choice among 5 ops, use 5
slots with a single 1 in the chosen slot (no false ordering implied — encoding
ops as 1,2,3,4,5 would falsely tell the model "avg_pool is 5× more than none").
6 edges × 5 = 30 slots. Path encoding: see D.5.3. Total 210.

## E.4 The models in our bake-off, in one line each

- **Linear/Ridge regression:** draws the best straight line; "ridge" adds a penalty
  on big weights to curb overfitting. Too rigid for our problem at large p.
- **Decision tree:** a flowchart of yes/no questions on features.
- **Random Forest (RF):** many trees, each trained on random subsets of
  data/features; average their answers. Robust but plateaued around 5/9 here.
- **Gradient boosting / XGBoost:** builds trees *sequentially*, each fixing the
  previous ones' mistakes. Powerful with lots of data — but **memorises tiny
  datasets** (our p=32 failure).
- **Gaussian Process (GP):** predicts with uncertainty by comparing to all training
  points via a similarity function. Good small-data behaviour but slow (it was
  competitive but not best, and 15–20× slower).
- **SVR:** see next.

## E.5 SVR with RBF kernel — how to explain it simply

**Support Vector Regression** fits the *flattest possible* function that passes
within a tolerance tube (width ε) of the training points. Points inside the tube
cost nothing; the model is anchored only by the few points on/outside the tube
(the **support vectors**).

The **RBF kernel** ("radial basis function") makes the function a sum of smooth
bumps centred on training points: prediction = weighted vote of training
architectures, where **similar architectures (close encodings) vote more**.

Two knobs:
- **C** — how hard to try to fit every point. Big C = fit tighter (risk
  overfitting); small C = smoother.
- **γ (gamma)** — the width of each bump. Big γ = narrow bumps = wiggly function;
  small γ = wide bumps = smooth function that generalises from few points.

**Why it beat XGBoost at p=32:** a tree can carve 32 points into 32 leaves
(= memorise). An RBF-SVR with small γ *cannot* wiggle that much — it's forced to
learn a smooth trend, which is exactly right when data is scarce.

**The ensemble:** SVR(C=20, γ=0.02) is smooth → wins at small p. SVR(C=10,
γ='scale') adapts more → wins at large p. Rank-averaging the two gives the best of
both at every p — that's what took us from 8/9 to 9/9 (CIFAR-10 p=1024 was the
last holdout: single SVRs gave 0.732–0.734 vs target 0.74; the ensemble: 0.741).

## E.6 Random search + predictor (what Task 4/5 step 2 actually is)

Plain random search = sample designs randomly, train them all, keep the best —
simple but expensive. **Predictor-guided random search** = sample 300 randomly,
let the predictor rank them (free), fine-tune only the top-10. The predictor
replaces ~290 expensive trainings per dataset. The check that this works: tau
between predicted and true accuracy on the 300 was 0.76–0.78.

## E.7 The loss we train with (Cosmic): BCE-with-logits, with an ignore mask

**Binary cross-entropy (BCE)** is the standard loss for yes/no questions: it
heavily punishes being *confidently wrong* per pixel. "With logits" = numerical
stability trick (loss takes raw scores, applies sigmoid internally).
**Our twist:** pixels labelled -9999 (unlabelable regions) are removed from the
loss — the network is neither rewarded nor punished there.

## E.8 AUROC — our evaluation metric (know this cold)

**AUROC = Area Under the ROC Curve.** Practical intuition:

> Pick one random cosmic-ray pixel and one random clean pixel. AUROC is the
> probability the model scores the cosmic-ray pixel higher.

- 1.0 = perfect separation, 0.5 = coin flip, <0.5 = systematically backwards.
- It's threshold-free: it evaluates the *ranking* of pixels, not one fixed cutoff.
- Why not plain accuracy? **Class imbalance**: ~99% of pixels are clean, so a model
  that says "no cosmic ray anywhere" would score ~99% accuracy while being useless.
  AUROC is immune to that trap.
- Our 0.976 val AUROC ⇒ 97.6% of ray/clean pixel pairs are ordered correctly.

## E.9 Transfer learning & fine-tuning (the "why" of everything)

**Transfer learning:** a network trained on task A has learned reusable feature
detectors (edges, blobs, textures in early layers). Starting task B from those
weights beats starting from random noise: faster convergence, better results with
little data.

**Fine-tuning:** continuing training of a pre-trained network on the new task,
usually with a *small* learning rate.

**Catastrophic forgetting:** if you fine-tune too aggressively, the gradients from
the new random head rip through the backbone and destroy the pre-trained features
before the head is even sensible. **Our defence = two phases:** first train only
the head (backbone frozen = used as a fixed feature extractor), then unfreeze all
with a 10× smaller LR.

**Details worth mentioning:**
- Cosmic is grayscale but NB201 backbones expect 3 channels → we repeat the
  channel 3× (`x.repeat(1,3,1,1)`).
- When loading pre-trained weights into our modified network we use
  `load_state_dict(strict=False)` — correct because the segmentation head is new
  (its weights legitimately don't exist in the checkpoint).
- "Which source dataset transfers best?" — in our preliminary run,
  **ImageNet16-120** (mean 0.968): plausibly because it's the hardest/most diverse
  source task, forcing more general features.

## E.10 Why the professor's targets are believable: the NASLib predictor benchmark

The target numbers (0.39/0.54/0.74…) correspond to published predictor benchmarks
(White et al., "How Powerful are Performance Predictors in NAS?"), which NASLib
implements. That's why "use ground truth from NASLib" matters — it puts our numbers
on the same footing as the literature.

---

# PART F — Cheat Sheet (memorise the night before)

**Search space:** 15,625 archs = 5 ops ^ 6 edges. Cell: 4 nodes, 6 edges. Ops:
none / skip / 1×1 conv / 3×3 conv / 3×3 avg-pool.

**Encoding:** 210 = 30 (6×5 one-hot) + 180 (paths: 5+25+25+125).

**Predictor:** rank-average of SVR(C=20, γ=0.02) + SVR(C=10, γ='scale').
Protocol: 2,000 held-out, 20 repeats, p ∈ {32…1024}. Ground truth: NASLib val acc.

**Predictor results (targets):** C10 0.439/0.596/0.741 (0.39/0.54/0.74) ·
C100 0.437/0.601/0.750 (0.39/0.56/0.74) · IN16 0.433/0.596/0.743 (0.40/0.56/0.74)
→ **9/9. XGBoost baseline: 5/9.**

**Selection:** 300 sampled (seed 99), top-10/dataset → 30 runs, 18 unique.
Tau on the 300: 0.758 / 0.776 / 0.760.

**Fine-tune recipe:** head-only 3 ep @ 1e-3 → full 5 ep @ 1e-4, Adam, batch 32,
loss = masked BCE-with-logits, metric = pixel AUROC.

**Task 3:** NinaPro arch-006186 → Cosmic: baseline 0.489 → P1 0.589 → **P2 0.976
val / 0.944 test** (371,473 params; head alone = 65 params; deepCR ref ≈ 0.97).

**Task 4/5 fine-tune (random init, 30/30 ok):** means C10 0.958 · C100 0.967 ·
IN16 0.968; best single 0.982 = `3-N-P-S-3-N` (ImageNet16-120 source).

**Cosmic data:** deepCR ACS-WFC, f435w GAL; 4,347/483/420 patches of 128×128;
ignore flag -9999.

**Weights saga:** small .pth = metrics only; 226 GB archive = dead link; real
weights = NATS-tss-full ~360 GB in 12 parts, quota-blocked for anonymous download;
need 18/15,625 files but split-tar can't be cherry-picked → Omni HPC cluster.

---

# PART G — Likely Viva Questions with Model Answers

**Q1. Why do we need a performance predictor at all?**
Training one NB201 network takes hours; there are 15,625. The predictor ranks all
of them in seconds from blueprints alone, so we spend GPU time only on the top
candidates.

**Q2. Why Kendall Tau and not MSE/accuracy of the predictor?**
We only use the predictor to *order* architectures and pick the top. A predictor
with a constant bias has terrible MSE but perfect ranking — and would still be
perfect for our use. Tau measures exactly the ordering quality.

**Q3. Why did XGBoost fail at p=32 and SVR succeed?**
Overfitting. Trees can partition 32 points into individually-memorised leaves; the
model fits noise and generalises poorly (tau ≈ 0.30). RBF-SVR is constrained to a
smooth function — with wide bumps (small γ) it can only learn the broad trend,
which is the right bias when data is scarce (tau ≈ 0.44).

**Q4. Was switching models allowed?**
The email specified the *metric and targets*, not the model. A predictor-model
comparison is standard methodology (the target numbers themselves come from a
predictor-comparison paper); we report the tree baseline (5/9) alongside SVR (9/9).

**Q5. What is rank averaging and why use it?**
Each ensemble member's predictions are converted to ranks (1st, 2nd, …) and the
ranks are averaged. It's scale-invariant (a model that predicts in a different
range isn't over/under-weighted) and it directly optimises ordering — which is what
tau measures.

**Q6. Why does path encoding help over one-hot?**
One-hot treats edges independently; but a network's behaviour depends on what ops
*compose along a path*. Path encoding gives the model explicit features for each
op-combination along the 4 input→output paths, so it can learn e.g. "conv→conv
chains are strong" directly.

**Q7. Why validation accuracy as ground truth, not test accuracy?**
The professor pointed to NASLib's ground truth (its API exposes val accuracy;
selecting on test would be test-set leakage). We verified both give near-identical
tau, so the conclusion is robust either way.

**Q8. Why sample 300 and pick 10 — why not predict on all 15,625 and take the top?**
The protocol simulates realistic *random search with a predictor* as specified.
(And in fact we do encode all 15,625, so we could rank all; but the experiment
tests the sampled-search setting the professor asked for.)

**Q9. Why did several architectures appear in the top-10 of multiple datasets?**
Architecture quality correlates strongly across these datasets — a genuinely good
cell (e.g. rich in 3×3 convs with a good skip structure) is good everywhere. That's
also why transfer between them is sensible. 30 selections → 18 unique nets.

**Q10. Why fine-tune instead of training from scratch?**
Pre-trained features (edges/textures) transfer; with a small dataset (4.3k patches)
and few epochs, starting from useful features converges faster and typically
better. Our preliminary random-init run still scored high (0.95+) — expected, since
the Cosmic task is learnable — but pre-trained initialisation is what the
assignment tests, and it should converge faster / at least as well.

**Q11. Why two-phase fine-tuning?**
The new head is random; if everything trains at once with a high LR, nonsense
gradients from the head can wreck the pre-trained backbone (catastrophic
forgetting). Phase 1 fixes the backbone and trains only the head; phase 2
fine-tunes everything at 10× lower LR.

**Q12. Why AUROC instead of pixel accuracy?**
Extreme class imbalance: predicting "no cosmic ray anywhere" gives ~99% pixel
accuracy but is useless. AUROC asks whether ray pixels score above clean pixels —
threshold-free and imbalance-robust.

**Q13. What do C and gamma do in your SVR?**
C = penalty for training errors (big C → tighter fit, less smoothing).
γ = inverse width of the RBF similarity bumps (small γ → smoother function).
We use a smooth SVR (γ=0.02) for small p and an adaptive one (γ='scale') for
large p, rank-averaged.

**Q14. Why not use a neural network as the predictor?**
With p as small as 32 labelled examples, small classical models are the right
capacity; a neural predictor would overfit worse than XGBoost did. (GP/BOHAMIANN
etc. exist in the literature; among our candidates SVR was best and cheapest.)

**Q15. Where are the pre-trained weights, and why didn't you just download them?**
They exist only in a ~360 GB 12-part archive (the older 226 GB link is dead); the
host throttles anonymous downloads and our disk couldn't hold it. Since only 18 of
15,625 files are needed — but a split tar can't be partially downloaded — we moved
download+extraction to the Omni cluster and bring back only ~30 small weight files.

**Q16. What's the ignore flag / why do some pixels have label -9999?**
Regions near saturated stars or chip edges can't be reliably labelled. They're
flagged and excluded from both loss and AUROC so the model is neither penalised nor
rewarded on unknowable pixels.

**Q17. What would you do next / with more time?**
Finish the pretrained-weight fine-tuning on Omni and compare vs random init;
try stronger predictors (GIN/graph neural networks over the cell DAG); use the
predictor inside evolutionary search (we prototyped this in task 3); test
transfer from the best *Cosmic-similar* source rather than only the three
benchmarks.

**Q18. Your baseline AUROC was 0.44 — below coin flip. Why?**
A random head on mismatched features can produce scores *anti-correlated* with the
truth; 0.5 is only the expectation over many random heads, any single one can land
below. It simply confirms "no training = no signal."

**Q19. What exactly is in the NB201 arch string?**
Six tokens `op~k` grouped by destination node: which operation each edge carries
and from which node it comes, e.g. `|3x3~0|+|1x1~0|3x3~1|+|none~0|3x3~1|3x3~2|`.
It's a complete blueprint of the cell.

**Q20. How do you know your predictor evaluation isn't lucky?**
Fixed held-out test set of 2,000 architectures never used in training; 20
independent random training draws per p; we report mean ± std (std was small,
0.004–0.05). Sampling seeds are fixed for reproducibility.

---

*Good luck tomorrow. Read Part A + F in the morning; if the professor drills
deep, everything is defensible from Parts D–E.*
