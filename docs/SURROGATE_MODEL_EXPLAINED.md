# Surrogate Models — What We Did and Why
### A focused companion to VIVA_PREP.md, covering the surrogate-model concept and its use in Task 3's evolutionary search

---

## 1. Where this fits in the project

"Surrogate model" is a general concept that actually runs through the **entire**
project — every performance predictor we ever built (Tasks 3, 4, 5) is technically
a surrogate model. But the term is used **explicitly and literally** in one
specific place: **Task 3's evolutionary search**, whose own file is titled
*"Evolutionary NAS with Surrogate Model."* This document covers both — the general
concept, and that specific implementation in detail.

```
Real evaluation (expensive)     Surrogate evaluation (cheap)
─────────────────────────       ─────────────────────────────
Build the network                Encode the blueprint (210 or 30 numbers)
Train it for hours on a GPU  →   Feed it through a small trained model
Measure its real accuracy        Get a predicted accuracy back — instantly
```

---

## 2. What a surrogate model is (from zero)

A **surrogate model** is any cheap, fast model that **stands in for** an
expensive, slow process, giving an approximate answer without actually running
the real thing.

**Analogy:** imagine you're testing thousands of car designs for fuel efficiency.
Physically building and road-testing every single design would take forever. So
instead, you build a computer simulation that *approximates* how efficient a
design would be, based on its blueprint alone — no need to actually manufacture
and drive the car. That simulation is a surrogate for the real road test.

In NAS, the "real, expensive process" is **training a neural network** (hours of
GPU time per architecture). A surrogate model replaces that with a fast, cheap
prediction from the architecture's blueprint alone — this is *exactly* the same
idea as the "performance predictor" already covered in VIVA_PREP.md (§E.2): **the
performance predictor is a surrogate model.** They are the same thing; "surrogate"
is simply the more general term used in the NAS literature, especially when the
predictor is embedded inside a larger search algorithm (as in this task).

---

## 3. The two places surrogate models appear in this project

| Where | What plays the role of "surrogate" | Purpose |
|---|---|---|
| **Tasks 4 & 5** (predictor + random search) | The XGBoost/SVR performance predictor | Score 300 sampled architectures instantly, without training any of them |
| **Task 3** (evolutionary search) | An XGBoost regressor, explicitly named "surrogate" in the code | Score *every generation* of mutated candidate architectures instantly, without training any of them |

The underlying idea is identical in both places. What differs is **how the
surrogate's scores are used to decide what to look at next**: in random search
(Tasks 4/5), the candidates are chosen randomly and the surrogate only ranks them
*after the fact*. In evolutionary search (Task 3), the surrogate's scores directly
**drive** which candidates get explored next — this tighter feedback loop is what
makes it worth calling out as its own distinct algorithm.

---

## 4. Task 3's evolutionary search with a surrogate — the full algorithm

This was implemented in `task3/evolutionary_search.py`, following the professor's
specified procedure exactly:

1. Initialise a population from **47 evaluated architectures** (real, known true
   accuracies, from `all/all/*.pickle.pbz2`, CIFAR-100 dataset).
2. Train an **XGBoost surrogate** on that population (blueprint → accuracy).
3. Rank the population by the surrogate's predicted score.
4. Select the **top-k parents** (k = 2).
5. **Mutate** each parent multiple times (20 mutations per parent) → a pool of
   candidate children.
6. **Evaluate candidates with the surrogate** — instantly, no GPU, no real
   training.
7. Add the **best 5** candidates into the population.
8. Re-rank the (now larger) population → go back to step 4.
9. Repeat until the population reaches **200** architectures.
10. Only *after* the search finishes, look up the **true** NAS-Bench-201 accuracy
    for the top-found architectures — to check whether the surrogate's judgement
    actually held up on architectures it never saw during training.

### 4.1 The mutation operator (how a "child" is created)

```python
def mutate(arch_str, rng, n_mutations=1):
    """Randomly replace 1 operation with a different one on a random edge."""
```

Given a parent architecture (a 6-edge blueprint), mutation picks a random edge and
swaps its operation for a *different* one of the remaining 4 choices. This is the
NAS equivalent of biological mutation: a small, random, local change to an
existing "genome" (architecture blueprint).

### 4.2 Why a surrogate is *essential* here, not just convenient

Evolutionary search needs to score candidates **every single generation** — with
32 iterations × up to 100 candidates evaluated per iteration (2 parents × 20
mutations each) in our run, that's **potentially thousands of evaluations**. If
each "evaluation" meant actually training a network for hours, this entire
algorithm would be completely infeasible. The surrogate is what turns "thousands
of evaluations" from an impossible GPU bill into a sub-second computation.

---

## 5. Our real results — and an honest, important nuance

### 5.1 Run settings and outcome

| Setting | Value |
|---|---|
| Initial population | 47 architectures (CIFAR-100) |
| Parents selected per iteration | 2 |
| Mutations per parent | 20 |
| Best candidates kept per iteration | 5 |
| Target population size | 200 |
| **Iterations actually run** | **32** |
| **Final population size** | **204** |

### 5.2 The surrogate's accuracy — two very different numbers

| Kendall Tau measured on... | Value | What it tells you |
|---|---|---|
| The 47 architectures the surrogate was **trained on** (in-sample) | **0.925** | How well the surrogate fits data it has already seen |
| **New** architectures discovered during the search (never in the training set) | **0.289** | How well the surrogate's judgement generalises to genuinely unseen candidates |

**This gap is the single most important, most honest finding from this
experiment**, and it's worth understanding rather than glossing over:

- **0.925 in-sample** looks excellent, but it's a somewhat misleading number — it
  mostly reflects that XGBoost fit the 47 training points well, which is expected
  and easy (this is the same small-p overfitting risk explained in VIVA_PREP.md
  §4.4, just observed from the *inside* this time).
- **0.289 on new architectures** is a much more modest, much more meaningful
  number — it reflects how well the surrogate's ranking held up on mutated
  children it never trained on. This is the number that actually matters for
  judging whether the evolutionary search made good decisions.
- Part of why 0.289 looks low: the *new* architectures found by mutating strong
  parents cluster tightly together in true accuracy (roughly 70.3%–73.5%, a narrow
  1200-point range) — Kendall Tau is naturally noisier and more fragile over a
  narrow, tightly-clustered set of values with only 10 samples, compared to the
  much wider spread used to measure Task 5's predictor (2,000 architectures
  spanning the *entire* accuracy range).

### 5.3 Did the search actually find something better?

| | Architecture | True accuracy |
|---|---|---|
| Best architecture already known (in the initial 47) | `3-3-3-S-3-3` | 73.503% |
| **Top-ranked result after the full search** | `3-3-3-S-3-3` (same one) | 73.503% |
| Best genuinely **new** architecture discovered | `3-1-3-S-3-3` | 73.147% |

**Honest reading:** the search's single top pick turned out to be the *same*
architecture that was already in the initial population — the search did not
surpass the best already-known design in this run. However, it *did* discover a
genuinely new, previously-unevaluated architecture (`3-1-3-S-3-3`) within **0.36
percentage points** of the best known design, and the surrogate's prediction for
it (73.149% predicted) was remarkably close to its true value (73.147% — an error
of just 0.002 points). That's a real, meaningful success even though it didn't set
a new record: it shows the surrogate can, at least sometimes, guide the search to
excellent *new* candidates with high precision — just not reliably enough to be
fully trusted (hence the 0.289 tau across all 10 new candidates, not just this
lucky one).

---

## 6. Why this matters for the viva — the honest narrative

If asked about this task, the strongest, most defensible answer is **not** "the
surrogate found a better architecture" — it's:

> *"Evolutionary search with a surrogate is a standard NAS technique that avoids
> real training at every generation by scoring candidates with a cheap predictor
> instead. In our run, the surrogate fit its own training population very well
> (Kendall Tau 0.925) but generalised much more modestly to genuinely new,
> mutated architectures (Kendall Tau 0.289) — a realistic illustration of exactly
> the overfitting risk that motivated Task 5's later work on a more robust
> predictor. The search still surfaced a genuinely new, high-quality architecture
> within 0.36 points of the best known design, with a highly accurate individual
> prediction — but did not beat the best-known architecture outright in this run."*

This is a more sophisticated and more credible answer than claiming a clean
victory, and it directly foreshadows *why* Task 5 later invested so much effort
into fixing the predictor's small-data generalisation problem (VIVA_PREP.md
§D.5.4) — this evolutionary-search experiment is early, first-hand evidence of
exactly that same weakness.

---

## 7. Quick answers for the viva

**Q: What is a surrogate model?**
A cheap, fast model that stands in for an expensive real evaluation — in NAS, it
predicts an architecture's accuracy from its blueprint instead of actually
training it.

**Q: Is the "performance predictor" from Tasks 4/5 also a surrogate model?**
Yes — they are the same concept. "Surrogate" is the general term; in this project
we usually called it "the predictor," except in Task 3's evolutionary search where
the code explicitly names it "surrogate."

**Q: Why does evolutionary search specifically need a surrogate (more so than random search)?**
Because it scores candidates repeatedly, every generation, potentially thousands
of times across the full run — real training at that volume would be completely
infeasible. Random search only needs to score its candidates once, at the end.

**Q: How well did the surrogate actually perform?**
Very well on its own training data (Kendall Tau 0.925) but much more modestly on
genuinely new, mutated architectures it had never seen (Kendall Tau 0.289) — an
honest illustration of the small-data generalisation problem, not a clean success
story.

**Q: Did the evolutionary search find a better architecture than what was already known?**
Not in this run — the top-ranked result was the same architecture already present
in the initial 47. It did discover a genuinely new candidate within 0.36
percentage points of the best known design, with a very accurate individual
prediction.

**Q: How does this connect to Task 5?**
This experiment is early, concrete evidence of the exact overfitting problem that
motivated Task 5's later investment in a more robust predictor (moving from
XGBoost to the SVR rank-ensemble) — the same failure mode (a tree-based model
fitting its small training set well but generalising poorly) shows up here too.
