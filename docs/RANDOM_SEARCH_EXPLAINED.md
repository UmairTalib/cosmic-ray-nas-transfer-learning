# Random Search in NAS — What We Did and Why
### A focused companion to VIVA_PREP.md, covering Step 2 of the pipeline in full detail

---

## 1. Where this fits in the project

Random search was **Step 2** of the pipeline (Tasks 4 and 5): after the performance
predictor (Step 1) was built and validated, we used it to search the NAS-Bench-201
space for good architectures to fine-tune on Cosmic.

```
Step 1: PREDICTOR        Step 2: RANDOM SEARCH         Step 3: FINE-TUNE
(build the appraiser)  → (use it to pick candidates)  → (train the winners)
```

This document explains random search itself, why it was the required method, exactly
how it was implemented, and the real results it produced.

---

## 2. What "random search" means in NAS (from zero)

**Neural Architecture Search (NAS)** needs a **search strategy** — a method for
deciding *which* of the many candidate architectures to actually evaluate, since
evaluating all of them (here, 15,625) for real would be far too expensive.

**Random search is the simplest possible strategy**: instead of using any clever
logic to decide what to try next, you just **pick candidates uniformly at random**
from the search space and evaluate those. No learning, no adaptation, no guided
exploration — pure chance, applied at a chosen sample size.

That sounds almost too simple to be useful — and normally it would be, if
"evaluate" meant *actually training* each randomly-picked network. But in our
pipeline, "evaluate" doesn't mean training — it means **asking the predictor**,
which is nearly free. That's what makes random search practical here: you can
afford to randomly sample a reasonably large pool (300) precisely *because*
scoring each one costs almost nothing.

### Random search vs. other strategies (for contrast)

| Strategy | How it picks candidates | Used in this project? |
|---|---|---|
| **Exhaustive search** | Evaluate literally everything | No — 15,625 real trainings is infeasible |
| **Random search** | Pick candidates uniformly at random | **Yes — Step 2, this document** |
| **Evolutionary search** | Start from a population, mutate the best performers, repeat | Yes — prototyped separately in Task 3 (see §7) |
| **Grid search** | Systematically try every combination in a structured sweep | No — not used |
| **Bayesian optimisation** | Use a probabilistic model to pick the *most informative* next candidate | No — not used |

---

## 3. Why random search specifically? (the honest answer)

Exactly as with the "why 300?" question — **this was a direct instruction from the
professor's email**, not a method we chose ourselves:

> *"...can be used to **randomly sample 300 networks** from the NAS-Bench-201 search
> space, predict the accuracy with the learned performance prediction model and
> fine-tune the best 10 predicted networks for each dataset..."*

So the answer to "why random search?" in the viva is simply: **it's what was asked
for.** That said, it's also a reasonable and defensible choice on its own merits:

- **It's an unbiased baseline.** Random sampling doesn't favour any region of the
  search space, so if the predictor successfully finds good architectures within a
  *random* sample, that's a meaningful demonstration — it isn't cherry-picked.
- **It directly tests the predictor.** The entire point of Step 2 is to show that a
  cheap predictor plus a simple search can find strong architectures without
  expensive evaluation. Random search is the cleanest possible test of that claim,
  because the search itself contributes nothing clever — all the "intelligence"
  comes from the predictor's ranking.
- **It's simple and reproducible.** One random draw, one fixed seed, no tuning
  knobs, no hyperparameters of the search method itself to justify.

---

## 4. Exactly how we implemented it

### 4.1 The mechanics (recap, made precise)

```python
rng = np.random.RandomState(99)                    # fixed seed -> reproducible
sample_idx = rng.choice(15625, size=300, replace=False)   # 300 unique indices
```

- **`RandomState(99)`**: a fixed seed. Re-running the script always draws the exact
  same 300 architectures — essential for reproducibility (see VIVA_PREP.md, Step
  13, for why this matters).
- **`replace=False`**: sampling *without* replacement — no architecture can be
  drawn twice into the same pool of 300.
- **One shared pool of 300** — not three separate draws. All three datasets'
  predictors score the *same* 300 candidates (see VIVA_PREP.md, Steps 14–16, for
  the full reasoning). This is what allows meaningful "this architecture is good
  across multiple datasets" conclusions later.

### 4.2 What happens to each of the 300

For each of the three datasets (CIFAR-10, CIFAR-100, ImageNet16-120):

1. Train that dataset's own predictor instance (the SVR rank-ensemble from
   VIVA_PREP.md §21–24) on **p = 1024** known architectures — the strongest
   configuration, since this is the real selection task, not a benchmark sweep.
2. Feed all **300** sampled architectures through that trained predictor.
3. Sort the 300 by predicted score, highest first.
4. Keep the **top 10**.

Repeated independently for all three datasets → 3 top-10 lists → 30 total
selections.

---

## 5. The real results

### 5.1 How good was the random sample, once ranked?

Before even looking at which architectures were picked, it's worth checking: did
the predictor's ranking of this fresh batch of 300 (which it never trained on
directly) actually track the true, known accuracy of those 300 architectures?

| Dataset | Kendall Tau (predicted vs. true, on the 300 samples) |
|---|---|
| CIFAR-10 | 0.758 |
| CIFAR-100 | 0.776 |
| ImageNet16-120 | 0.760 |

These are notably **higher** than the p=1024 benchmark numbers from Step 1
(0.741 / 0.750 / 0.743) — expected, since Kendall Tau computed over a smaller,
randomly-drawn set of 300 tends to have higher variance and can land above or
below the large-scale benchmark figure by chance. Either way, all three confirm
the predictor generalises well to an unseen batch.

### 5.2 The actual selections: 30 picks → 18 unique architectures

| Architecture | Encoding | Picked for |
|---|---|---|
| 000137 | S-3-P-S-3-3 | CIFAR-10 |
| 002352 | 3-1-3-P-3-1 | CIFAR-100 |
| 002658 | 3-S-S-N-3-3 | CIFAR-10, CIFAR-100 |
| 002870 | 3-3-1-S-1-S | CIFAR-100, ImageNet16-120 |
| 003275 | 3-3-N-1-3-S | ImageNet16-120 |
| 003990 | 3-1-3-3-N-1 | ImageNet16-120 |
| 005115 | 3-3-1-1-3-3 | ImageNet16-120 |
| 005204 | 1-S-3-S-1-3 | CIFAR-10, ImageNet16-120 |
| 007335 | 3-3-1-3-1-1 | CIFAR-100, ImageNet16-120 |
| 007632 | 3-3-3-3-3-1 | CIFAR-100 |
| 008002 | 3-3-3-S-P-1 | CIFAR-10 |
| **008993** | **S-3-3-3-3-1** | **all 3 datasets** |
| **009448** | **1-S-S-3-3-3** | **all 3 datasets** |
| 010004 | N-3-S-3-P-3 | CIFAR-10 |
| 010436 | 3-1-3-1-3-3 | CIFAR-100, ImageNet16-120 |
| 014255 | 1-S-N-3-3-3 | CIFAR-10, CIFAR-100 |
| **014289** | **3-3-S-P-1-3** | **all 3 datasets** |
| 014410 | 3-N-N-1-3-N | CIFAR-10 |

Arithmetic check: 8 architectures picked once + 7 picked twice + 3 picked three
times = 8(1) + 7(2) + 3(3) = 8 + 14 + 9 = **30** ✓, collapsing to **18 unique**
designs. These 18 are exactly what moved on to Step 3 (fine-tuning).

---

## 6. What random search does *not* do — and why that's fine here

It's worth being clear-eyed about random search's real limitation, in case the
viva probes it: **a random sample of 300 out of 15,625 (≈1.9%) has no guarantee of
containing the single best architecture in the entire space.** Unlike evolutionary
search (§7) or Bayesian optimisation, random search never "learns" from earlier
draws to bias later ones toward better regions — every draw is independent of
architecture quality.

This is an acceptable trade-off here for two reasons:

1. **The professor's protocol only asks for "a good architecture found cheaply,"
   not "the provably best architecture in the whole space."** Random search
   answers exactly the question being asked.
2. **NAS-Bench-201 architectures are highly redundant in quality** — many
   different edge combinations produce similarly strong cells (e.g. any design
   with plenty of 3×3 convolutions and few wasted `none` operations tends to
   perform well). A 300-sample draw is very likely to contain *several* strong
   candidates even without any guided search, which the results in §5.2 confirm
   (see how many architectures repeated across all three predictors' top-10s).

---

## 7. Random search vs. the evolutionary search we also built (Task 3)

Task 3 prototyped a different, more sophisticated strategy: **evolutionary search
with a surrogate model** (see VIVA_PREP.md §D.3.3). It's worth contrasting the two
directly, since both appear in this project and a viva question could easily ask
"why not use evolutionary search for the real pipeline too?"

| | Random search (Step 2, used for the real pipeline) | Evolutionary search (Task 3 prototype) |
|---|---|---|
| How candidates are chosen | Uniformly at random, one draw | Start from a population; repeatedly mutate the best performers and keep improving |
| Uses feedback from earlier candidates? | No — every draw is independent | Yes — later generations are biased toward what already worked |
| Simplicity | Very simple, one line of code | More moving parts: selection, mutation, generational loop |
| Risk | May miss stronger regions of the space entirely | Can get stuck favouring a narrow region ("local optimum") |
| Why used here | Directly specified by the professor's email for the final pipeline | Explored earlier as a NAS-strategy prototype, not carried into the final selection step |

Both approaches rely on the *same* underlying idea — using a cheap surrogate/
predictor instead of real training to score candidates — they just differ in how
candidates are chosen to be scored.

---

## 8. Quick answers for the viva

**Q: What is random search?**
A NAS search strategy that evaluates candidates chosen uniformly at random from
the search space, with no adaptive logic guiding the sampling.

**Q: Why 300 samples, and why random (not systematic)?**
Both were directly specified in the professor's email requirements — not
parameters we tuned ourselves.

**Q: Is one pool of 300 shared across all three datasets, or three separate pools?**
One shared pool (fixed seed 99), scored three times by three separately-trained,
dataset-specific predictors.

**Q: Does random search guarantee finding the best architecture?**
No — it has no mechanism to bias sampling toward stronger regions of the search
space. It only guarantees a reasonable chance of finding *good* architectures,
proportional to how common they are in the space and how large the sample is.

**Q: How do you know the random search actually worked?**
The predictor's ranking of the 300 sampled architectures still tracked their true
accuracy well (Kendall Tau 0.758 / 0.776 / 0.760) — meaning the predictor
generalised to unseen candidates, not just the data it trained on. And 3
architectures were independently selected by all three dataset-specific
predictors, a signal that genuinely strong, transferable designs exist within the
sample.

**Q: How does this differ from the evolutionary search also mentioned in the project?**
Evolutionary search (Task 3) adapts over generations, mutating the best-found
architectures to bias later search toward promising regions. Random search (Step
2, used for the real pipeline) draws every candidate independently with no such
adaptation — simpler, and exactly what the assignment specified.
