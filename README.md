# Reranking Exemplars by Utility for Multimodal In-Context Learning

This study asks whether a learned reranker can select a better one-shot
demonstration than CLIP nearest-neighbor retrieval for fine-grained image
classification.

## 1. Background

In-context learning (ICL) lets a model adapt to a task from examples placed in
its prompt. Performance depends heavily on which examples are selected:
retrieval based on semantic similarity is useful, but the nearest example is 
not necessarily the one that most improves the model's accuracy (see Oracle 
Upper Bound).

This work is chiefly inspired by Hashimoto et al.,
[*Take One Step at a Time to Know Incremental Utility of Demonstration*](https://arxiv.org/abs/2311.09619).
They train a reranker model on model-derived marginal utility labels, and show
that selection from reranked candidates improves 1-shot performance one a number
of different text-based tasks (including classification) on the CLIP nearest-neighbor
baseline. This study extends that idea to multimodal classification, where utility must
account for the query image, exemplar image, and exemplar label.

## 2. Goals

1. Measure the benefit and limitations of CLIP-retrieved one-shot examples.
2. Estimate the accuracy ceiling available from reranking CLIP's top-30 candidate pool.
3. Generate utility data with Idefics2, simulating the test-time forward pass on <exemplar image, exemplar label, query image> triplets.
3. Train a reranker that predicts which exemplar most improves Idefics2 classification.
4. Evaluate the learned selector on held-out queries over the full 200-class label space.

## 3. Experimental design

### Task and splits

- **Dataset:** CUB-200-2011, containing 200 visually similar bird species.
- **VLM:** Idefics2-8B.
- **Task:** 0-shot and 1-shot image classification.
- **Retrieval pool:** labeled images from the train split.
- **Teacher/model-selection queries:** train and validation splits only.
- **Final evaluation:** held-out test split; no test query is used to generate
  teacher targets or select a model.

CLIP retrieves the 30 nearest candidate exemplars. SigLIP image-to-text
similarity constructs query-specific hard label sets containing the true class
and the most confusable alternatives.

Two scoring views are used:

- **Restricted K-way scoring** reduces noise and teacher-generation cost by
  evaluating only hard candidate labels.
- **Full-label scoring** evaluates all 200 labels under one fixed,
  option-free prompt and is the primary protocol for final claims.

### Utility definition

Let `z_c(q,e)` be Idefics2's mean-token log-likelihood score for class `c`,
given query `q` and exemplar `e`. The primary utility signal is the
true-vs-best-wrong margin:

```text
m_K(q,e) = z_true(q,e) - max z_c(q,e), for c in the K-way set and c != true
```

A positive margin means the true class is the argmax. Larger positive margins
indicate more confident correct predictions. Raw score vectors are retained so
probability, direct margin, incremental margin, and alternative temperatures
can be compared without rerunning Idefics2.

The design addresses four known pitfalls:

| Pitfall | Design response |
|---|---|
| Raw true-label probability ignores competing labels | Normalize over a fixed closed label set |
| Higher true-label probability may still leave the class below the argmax | Optimize the true-vs-strongest-wrong score margin |
| Difficult queries produce smaller absolute target differences | Rank exemplars within each query rather than comparing utilities across queries |
| Image-only inputs can collapse to visual similarity | Include the exemplar's label as a required reranker input |

The pairwise objective ignores exemplar pairs whose teacher-margin difference
is at most 0.02 by default; this threshold will be confirmed on training data
and selected without using the held-out test split.

### Choosing the teacher label-set size

Scoring all 200 classes for every query-exemplar pair is expensive. A
50-query validation audit compared restricted margins with full-200 margins:

| K | Relative cost | Spearman | Kendall | Top-exemplar agreement | Full accuracy gap | Mean margin regret |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 8% | 0.895 | 0.838 | 86% | 0 points | 0.0176 |
| 24 | 12% | 0.935 | 0.896 | 90% | 0 points | 0.0107 |
| 32 | 16% | 0.964 | 0.926 | 92% | 0 points | 0.0070 |
| 48 | 24% | 0.976 | 0.952 | 94% | 0 points | 0.0034 |
| 200 | 100% | 1.000 | 1.000 | 100% | 0 points | 0 |

K=32 is used for teacher generation: it costs 16% of full scoring, agrees with
the full-space top exemplar on 92% of audited queries, and reduces mean margin
regret by about 60% relative to K=16. Final evaluation remains full-200.

## 4. Baseline evaluation results

Unless noted otherwise, results below use all 1,767 held-out test queries.
Top-1 and top-5 indicate whether the true class ranks first or within the first
five classes under the fixed 200-label score vector.

### Zero-shot versus CLIP one-shot

| Condition | Top-1 | Top-5 | Median true rank | Mean true rank |
|---|---:|---:|---:|---:|
| Zero-shot | 13.75% | 23.20% | 29 | 48.46 |
| CLIP one-shot | 50.42% | 68.14% | 1 | 16.51 |

A single CLIP-retrieved exemplar improves top-1 accuracy by **36.67 percentage
points**.

### Dependence on same-class retrieval

| Retrieved exemplar | Queries | Zero-shot | One-shot | One-shot top-5 |
|---|---:|---:|---:|---:|
| Same class | 983 | 17.50% | 82.60% | 95.42% |
| Different class | 784 | 9.06% | 10.08% | 33.93% |

About 91.1% of correct one-shot predictions use a same-class exemplar. Across
all queries, the one-shot prediction equals the exemplar class 71.08% of the
time. When CLIP retrieves the wrong class, Idefics2 copies that class 56.63% of
the time, compared with predicting it only 4.08% of the time zero-shot.

These results show that Idefics2 benefits strongly from retrieval, but also
exhibits a substantial exemplar-copying bias.

### Score-distribution diagnostics

The “probability” below is a softmax-normalized mass over the 200
mean-token-log-likelihood class scores, not the literal probability of
generating the complete class string.

| Diagnostic | Zero-shot | CLIP one-shot |
|---|---:|---:|
| Mean true-class probability | 2.33% | 7.85% |
| Mean top-1/top-2 score gap | 0.256 | 0.765 |
| Mean entropy | 4.567 | 4.443 |
| Mean effective classes | 97.1 | 87.9 |

The true class receives measurably more mass than random chance in both
conditions. The exemplar primarily increases the true-class score and decision
margin; the overall distribution narrows only moderately.

### Accuracy as the hard-label set widens

| Candidate labels | Zero-shot | CLIP one-shot | One-shot gain |
|---:|---:|---:|---:|
| 4 | 38.94% | 62.99% | +24.05 pp |
| 8 | 27.33% | 57.44% | +30.11 pp |
| 12 | 23.54% | 55.35% | +31.81 pp |
| 16 | 20.71% | 53.76% | +33.05 pp |
| 200 | 13.75% | 50.42% | +36.67 pp |

Accuracy decreases monotonically as plausible distractors are added. K=16 is
already close to the full-label result, indicating that SigLIP concentrates
most confusable classes into a small query-specific set.

### Oracle upper bound

The oracle evaluates every exemplar in the same CLIP top-30 pool and chooses
the one with the largest true-vs-best-wrong margin. On the oracle-evaluated
subset:

| K | CLIP top-1 | 30-candidate oracle | Gain |
|---:|---:|---:|---:|
| 4 | 63.16% | 92.53% | +29.37 pp |
| 8 | 58.00% | 89.89% | +31.89 pp |
| 12 | 56.11% | 88.74% | +32.63 pp |
| 16 | 54.74% | 87.79% | +33.05 pp |

The CLIP values differ slightly from the complete-test table because this
oracle analysis uses the subset with completed exhaustive candidate scores.
All gains above compare paired queries within that subset.

At K=16, oracle accuracy grows as more CLIP candidates become available:

| Candidates available | Oracle accuracy |
|---:|---:|
| 1 | 54.74% |
| 2 | 65.68% |
| 3 | 70.84% |
| 5 | 76.32% |
| 10 | 82.42% |
| 20 | 86.21% |
| 30 | 87.79% |

Returns diminish beyond 20 candidates, so CLIP similarity contains useful
signal even though its top-1 choice is often suboptimal.

Same-class exemplars still dominate the oracle at K=16:

- Accuracy when the pool contains the true class: **90.23%**
- Accuracy when the pool omits the true class: **10.34%**
- Oracle-selected exemplars from the true class: **92.63%**
- Mean oracle true-class rank: **1.31**
- Mean oracle top-1/top-2 margin: **+1.083**

Different-class examples occasionally help: among pools without the true
class, 51.7% contain a successful exemplar at K=4, falling to 10.34% at K=16.

## 5. Reranker model architecture [WIP]

### Input representation

The minimal model receives pooled SigLIP embeddings for the query image,
exemplar image, and exemplar label. SigLIP supplies a shared image/text space
and is related to Idefics2's initial vision backbone, but it is not the exact
Idefics2 visual representation after multimodal adaptation.

CLIP similarity, full CLIP embeddings, retrieval rank, and explicit
image-label similarities are added only as separate ablations. The model never
receives the query's true label or a query/exemplar same-class indicator.

### MLP vs. transformer

The first architecture is a compact interaction MLP over the three SigLIP
vectors. It explicitly supplies query/exemplar, query/label, and exemplar/label
products and differences. This is the sample-efficient baseline.

The controlled transformer comparison uses the same inputs and objective as a
four-token sequence:

```text
[SCORE] [QUERY_IMAGE] [EXEMPLAR_IMAGE] [EXEMPLAR_LABEL]
```

`[SCORE]` is a learned readout vector shared across all examples, not the
teacher utility. Through self-attention it gathers information from the three
content tokens, and its final state is mapped to one candidate score. Mean
pooling the content-token states is a simpler control. Architecture selection
uses query-disjoint validation selector accuracy and full-space margin regret,
not training loss alone.

### Attention: pooled tokens vs. image patches

Self-attention across the four pooled tokens tests learned fusion but cannot
compare spatial regions within the images. Patch-level cross-image attention
could capture fine-grained bird markings, but requires a new feature artifact
and considerably more data, storage, and compute. It will be attempted only if
pooled models underfit and error analysis indicates missing spatial evidence.

### Loss: pointwise vs. pairwise

The first learned baseline follows the reference paper: pointwise soft-label
logistic regression on direct closed-set true-class probability. With the MLP
held fixed, experiments then compare incremental probability and bounded
margin targets.

Pairwise ranking is a proposed improvement evaluated afterward using the same
selected utility target, so only the loss changes. It optimizes within-query
order directly, but loses absolute calibration, creates quadratically many
candidate pairs, and needs a target-specific near-tie threshold. Pairs are
never formed across queries; raw margin ranking remains a separate target
ablation.

### Experimental controls and VLM scope

Inputs, target, loss, and architecture are changed one factor at a time. The
resulting reranker is Idefics2-specific because its utility labels come from
Idefics2; changing the target VLM normally requires regenerated teacher targets
and reranker fine-tuning or retraining.

See [RERANKER_ARCHITECTURE_PLAN.md](RERANKER_ARCHITECTURE_PLAN.md) for the
complete decision log, model dimensions, ablation order, and implementation
status.

## 6. Reranker evaluation results

**Results pending.** No trained-reranker accuracy is reported yet.

The final evaluation will use the held-out test split, the same fixed CLIP
top-30 candidate pools, and full-200 label scoring:

| Selector | Full-200 top-1 |
|---|---:|
| Zero-shot | 13.75% |
| CLIP top-1 | 50.42% |
| Learned reranker | Pending |
| 30-candidate oracle | Pending full-200 result |

The primary metric is full-200 top-1 accuracy. Secondary metrics are top-5
accuracy, mean reciprocal rank, full-space margin regret, agreement with the
oracle, and performance split by whether the retrieval pool contains a
same-class exemplar. Restricted K-way results will be reported as diagnostics,
not as the main claim.

## Reproducing the study

Install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Prepare the CUB split and embedding artifacts:

```bash
python -m scripts.create_image_split --dataset cub_200
python -m scripts.build_clip_embeddings \
  --dataset cub_200 \
  --image-split-path data/cub_200/image_split.json
python -m scripts.build_siglip_embeddings \
  --dataset cub_200 \
  --image-split-path data/cub_200/image_split.json
```

Run the completed study stages:

```bash
# Full held-out baseline
python -m scripts.evaluate_full_label_baselines scoring.num_queries=null

# Oracle; provide the completed baseline artifact
python -m scripts.evaluate_full_label_oracle \
  input.baseline_results_path=outputs/evals/full_label_baselines/<run>.pkl \
  oracle.scope=all

# Validate K=32 and generate reranker supervision
python -m scripts.audit_label_space_k
python -m scripts.generate_reranker_teacher_data
```

Configurations live in [`configs/`](configs/). Run `pytest` for the GPU-free
unit suite.
