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

### Best configuration so far

The winning recipe is a compact interaction MLP over pooled SigLIP embeddings
(query image, exemplar image, exemplar label), trained on the raw margin
target with the hybrid listwise-pairwise loss (`hybrid_listwise_weight=0.1`,
`pairwise_min_target_gap=0.02`), hidden width 256, dropout 0.1–0.2, learning
rate 3e-4, and weight decay 1e-3–1e-4. It beat every alternative tried: a
matched frozen-Idefics2-state probe, a pooled transformer, cross-candidate
attention, image-patch cross-encoding, correctness-only pairwise loss, and
narrower widths (128).

At K=32 validation, this configuration reaches **~74.3–74.9% restricted
selection accuracy** against an **85.1% pool-oracle ceiling** (the accuracy of
always picking the best of the 30 CLIP candidates in hindsight). Error
analysis attributes most of the remaining gap to generalization rather than
model capacity or objective choice — the model already tracks its own
training-set oracle closely, but on held-out queries it frequently picks an
exemplar from the wrong class rather than just the wrong image within the
right class.

| Selector | Accuracy | Scope |
|---|---:|---|
| Zero-shot | 13.75% | Full-200, held-out test |
| CLIP one-shot (top-1 retrieval) | 50.42% | Full-200, held-out test |
| Learned reranker (best config, one-shot) | ~74.3–74.9% | Restricted K=32, validation |
| 30-candidate pool oracle | 85.1% | Restricted K=32, validation |

Each stage adds information the previous one lacks: retrieval alone, then a
learned choice among 30 retrieved candidates, then the hindsight-best choice
among those same candidates. The reranker and oracle rows use restricted K=32
validation scoring rather than the full-200 held-out test protocol, so they
are not yet directly comparable to the zero-shot/CLIP rows above them — see
[section 6](#6-reranker-evaluation-results) for the pending full-200 test
comparison on the same scale.

### Design decisions

- **Input representation** — pooled SigLIP embeddings for query image,
  exemplar image, and exemplar label; CLIP similarity/embeddings, retrieval
  rank, and explicit image-label similarities were tested only as separate
  ablations and did not replace this minimal set, and the model never sees
  the query's true label or a same-class indicator.
- **Architecture** — a compact interaction MLP with explicit
  query/exemplar, query/label, and exemplar/label products and differences
  outperformed a four-token pooled transformer (`[SCORE] [QUERY_IMAGE]
  [EXEMPLAR_IMAGE] [EXEMPLAR_LABEL]`) and a patch-level `visual_token_cross_encoder`
  that attends over frozen Idefics2 visual/label tokens, judged by
  query-disjoint validation accuracy and margin regret rather than training loss.
- **Target** — raw true-vs-best-wrong margin beat direct/incremental
  output probability and bounded-margin transforms, since margin alone
  guarantees every correct candidate outranks every incorrect one.
- **Loss** — pairwise Bradley-Terry ranking on the margin target, hybridized
  with a small multiple-positive listwise term (`hybrid_listwise_weight=0.1`)
  so training targets top-1 selection directly rather than only pointwise
  probability calibration; pure correctness-crossing pairwise loss (dropping
  within-status margin structure) hurt generalization and was rejected.
- **Pairwise near-tie filter** — `pairwise_min_target_gap=0.02` excludes
  candidate pairs with near-identical margins from the loss; a follow-up
  sweep (0.02/0.05/0.1) found no setting that reliably improved on this
  default, so it was kept.
- **Regularization** — a grid search over dropout, hidden width, learning
  rate, and weight decay confirmed width 256 and learning rate 3e-4 as clear
  wins over width 128 and 1e-3, while dropout 0.1 vs. 0.2–0.3 is a minor
  accuracy/calibration tradeoff rather than a clear winner.
- **Frozen-Idefics2-state probe** — a parameter-matched MLP over a single
  cached pre-answer Idefics2 state was tested as an alternative to the pooled
  SigLIP representation; it underperformed on this margin/listwise objective,
  though it showed complementary errors, so it was kept as a diagnostic
  rather than adopted.
- **Experimental controls and VLM scope** — inputs, target, loss, and
  architecture are changed one factor at a time, and the reranker remains
  Idefics2-specific since its utility labels come from Idefics2; targeting a
  different VLM would require regenerating teacher targets and retraining.

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

# Paper-aligned first reranker: minimal SigLIP interaction MLP, direct
# true-class probability target, and pointwise soft-label BCE.
python -m scripts.train_reranker \
  data.artifact_path=/path/to/reranker_teacher_data.pkl
```

To expand a completed ranked M=30 artifact to M=100 without rescoring ranks
1–30 or the zero-shot condition, use explicit prefix expansion and publish to
a new dataset. The generator reconstructs the top-100 CLIP ranking and requires
every stored M=30 candidate array to match its exact prefix before launching a
worker:

```bash
python -m scripts.generate_reranker_teacher_data \
  dataset.image_split_path=/path/to/image_split.json \
  retrieval.candidate_pool_size=100 \
  limits.resume_from=/path/to/m30/reranker_teacher_data.pkl \
  limits.allow_candidate_pool_expansion=true \
  output.save_dir=/path/to/reranker-teacher-m100 \
  output.kaggle_dataset=juandatan/cub-200-reranker-teacher-data-m100
```

The original dataset is never modified. Partial M=100 checkpoints contain the
untouched M=30 prefixes for unfinished queries and upload periodically to the
new dataset. After a restart, point `limits.resume_from` at the latest M=100
artifact; its completed queries are skipped and unfinished queries continue at
rank 31.

Train first on the top 50 candidates from the completed M=100 artifact:

```bash
python -m scripts.train_reranker \
  data.artifact_path=/path/to/m100/reranker_teacher_data.pkl \
  data.max_candidates=50 \
  data.target=margin \
  model.architecture=interaction_mlp \
  objective.name=hybrid_listwise_pairwise \
  objective.hybrid_listwise_weight=0.1 \
  objective.pairwise_min_target_gap=0.02 \
  objective.pairwise_score_temperature=1.0 \
  experiment.seed=40 \
  optimization.learning_rate=0.0003 \
  optimization.weight_decay=0.0001 \
  optimization.epochs=100
```

Set `data.max_candidates=100` for the follow-up. Both training and validation
use the same ranked prefix; incomplete generator checkpoints are rejected.

Useful controlled overrides are:

```bash
# Bounded-margin target with the same model and pointwise objective
python -m scripts.train_reranker \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.target=bounded_margin data.target_temperature=1.0 \
  experiment.name=cub_200_siglip_mlp_bounded_margin

# Four-token pooled Transformer with the original probability target
python -m scripts.train_reranker \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  model.architecture=pooled_transformer \
  experiment.name=cub_200_siglip_transformer_probability

# Pairwise raw-margin ablation
python -m scripts.train_reranker \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.target=margin objective.name=pairwise \
  experiment.name=cub_200_siglip_mlp_pairwise_margin

# Multiple-positive listwise correctness objective. The margin target is kept
# for validation diagnostics; training uses the stored teacher-correct mask.
python -m scripts.train_reranker \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.target=margin objective.name=listwise_correctness \
  experiment.name=cub_200_siglip_mlp_listwise_correctness

# Margin-pairwise baseline plus a conservative correctness-listwise term
python -m scripts.train_reranker \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.target=margin objective.name=hybrid_listwise_pairwise \
  objective.hybrid_listwise_weight=0.25 \
  experiment.name=cub_200_siglip_mlp_hybrid_listwise_pairwise

# Accuracy-aligned alternative: construct pairs only across the teacher
# correctness boundary. Queries with no correct candidate contribute zero;
# set correctness_margin_aux_weight=0.1 to retain raw-margin ordering as a
# small auxiliary, including on those all-incorrect queries.
python -m scripts.train_reranker \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.target=margin objective.name=correctness_crossing_pairwise \
  objective.correctness_margin_aux_weight=0.0 \
  experiment.seed=40 optimization.epochs=100

# Proportional class-stratified learning curve. Use 0.25 and 0.5, then compare
# against the existing full-data run. Train and validation selection accuracy
# are evaluated deterministically after every epoch by default.
python -m scripts.train_reranker \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.target=margin objective.name=correctness_crossing_pairwise \
  data.train_fraction=0.25 experiment.seed=40 optimization.epochs=100
# For especially expensive architectures, disable the extra train evaluation
# pass with logging.evaluate_train_metrics=false.

# Contextualize the interaction-MLP representations across the candidate pool
python -m scripts.train_reranker \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.target=margin objective.name=hybrid_listwise_pairwise \
  objective.hybrid_listwise_weight=0.1 \
  model.architecture=cross_candidate_attention

# First cache teacher-equivalent Idefics2 visual and exemplar-label token
# states. By default extraction uses only the first 4.99 GB shard of the
# official AWQ checkpoint; that shard contains the unquantized vision model,
# connector, and input embedding table. The quantized language model is never
# downloaded or instantiated, so AutoAWQ is not required. The builder verifies
# the source contract against the teacher architecture at runtime. Visual
# tokens use resumable per-token INT8 storage with FP16 scales (~2.45 GiB).
python -m scripts.build_reranker_visual_token_cache \
  dataset.teacher_artifact_path=/path/to/reranker_teacher_data.pkl \
  dataset.image_split_path=/path/to/image_split.json \
  output.cache_dir=/path/to/reranker_visual_tokens

# For a direct, non-quantized source comparison, override both settings. The
# feature-only loader needs the first two original shards (~9.6 GB), rather
# than the complete 33.6 GB checkpoint, and stores ~4.9 GiB of FP16 tokens:
#   model.feature_source_model=HuggingFaceM4/idefics2-8b output.dtype=float16

# Candidate-independent patch/token cross-encoder. A smaller query-group batch
# and AMP are recommended because each query contains 30 visual-token pairs.
python -m scripts.train_reranker \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.visual_token_cache_path=/path/to/reranker_visual_tokens \
  data.batch_size=8 data.target=margin \
  objective.name=hybrid_listwise_pairwise \
  objective.hybrid_listwise_weight=0.1 \
  model.architecture=visual_token_cross_encoder \
  optimization.amp=true optimization.epochs=100

# Frozen-Idefics2 cross-encoder probe. The first command runs the frozen LM
# once per exemplar/query pair and stores its final pair-conditioned state in
# resumable per-vector INT8 form (~1.15 GiB / 1.23 GB). It reuses the
# visual-token cache to avoid re-encoding images. Neither command exposes K=32
# labels or the query ground truth to the model. By default it loads the
# official pre-quantized AWQ checkpoint rather than downloading the 33.6 GB
# original checkpoint. Install the project with its AWQ extra so GPTQModel and
# dependencies such as PyPcre, threadpoolctl, and tokenicer are resolved as one
# consistent set. This cache consumes existing visual tokens and does not use
# torchvision itself, although GPTQModel currently imports torchvision while
# eagerly registering unrelated model definitions:
#   pip install --no-deps torchvision==0.23.0 \
#     --index-url https://download.pytorch.org/whl/cu128
#   pip install --upgrade-strategy only-if-needed -e ".[awq]"
# The AWQ extra pins torchao 0.16 and torchvision 0.23 to PyTorch 2.8.
# Probe loading explicitly uses GPTQModel's GEMM Triton backend, avoiding the
# Marlin JIT path and its dependency on a matching system nvcc. A slower
# fallback is available with model.awq_backend=torch_awq.
python -m scripts.build_frozen_idefics2_probe_cache \
  dataset.teacher_artifact_path=/path/to/reranker_teacher_data.pkl \
  dataset.visual_token_cache_path=/path/to/reranker_visual_tokens \
  output.cache_dir=/path/to/frozen_idefics2_probe_cache

# Publish only after metadata confirms that every pair is complete. The upload
# staging area uses hard links, so it does not duplicate the ~1.23 GB cache.
python -m scripts.upload_frozen_idefics2_probe_cache_to_kaggle \
  --cache-dir /path/to/frozen_idefics2_probe_cache \
  --dataset-name owner/cub-200-frozen-idefics2-probe-cache

# OpenAI CLIP is not needed above. Install it separately only when regenerating
# CLIP embeddings (the compatible torchvision dependency is already present):
#   pip install -e ".[clip]"

# Paper-aligned direct probability: exp(mean-token log likelihood), without a
# softmax over the K=32 output-label set.
python -m scripts.train_frozen_idefics2_probe \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.probe_cache_path=/path/to/frozen_idefics2_probe_cache \
  data.target=mean_token_probability

# Nonlinear-accessibility ablation over the identical cached states and target.
python -m scripts.train_frozen_idefics2_probe \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.probe_cache_path=/path/to/frozen_idefics2_probe_cache \
  data.target=mean_token_probability \
  model.architecture=layernorm_mlp model.hidden_dim=256 model.dropout=0.1 \
  optimization.learning_rate=0.0003

# Same frozen-state MLP and probability target, but optimize within-query
# exemplar preferences instead of independent probability calibration.
python -m scripts.train_frozen_idefics2_probe \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.probe_cache_path=/path/to/frozen_idefics2_probe_cache \
  data.target=mean_token_probability \
  model.architecture=layernorm_mlp model.hidden_dim=256 model.dropout=0.1 \
  objective.name=pairwise objective.pairwise_min_target_gap=0.02 \
  optimization.learning_rate=0.0003

# Preserve probability calibration and add ranking as a 0.1-weight auxiliary.
python -m scripts.train_frozen_idefics2_probe \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.probe_cache_path=/path/to/frozen_idefics2_probe_cache \
  data.target=mean_token_probability \
  model.architecture=layernorm_mlp model.hidden_dim=256 model.dropout=0.1 \
  objective.name=pointwise_pairwise objective.hybrid_pairwise_weight=0.1 \
  objective.pairwise_min_target_gap=0.02 \
  optimization.learning_rate=0.0003

# Separate top-weighting ablation: lower temperatures focus pairwise weight on
# pairs containing candidates with the highest teacher probabilities.
# Add: objective.pairwise_teacher_weight_temperature=0.05

# Exact frozen-state comparison with the leading pooled-MLP recipe: raw margin
# pairwise ranking plus a 0.1-weight correctness-listwise auxiliary.
python -m scripts.train_frozen_idefics2_probe \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.probe_cache_path=/path/to/frozen_idefics2_probe_cache \
  data.target=margin \
  model.architecture=layernorm_mlp model.hidden_dim=450 model.dropout=0.1 \
  objective.name=hybrid_listwise_pairwise \
  objective.hybrid_listwise_weight=0.1 \
  objective.pairwise_min_target_gap=0.02 \
  optimization.learning_rate=0.0003

# Incremental-probability probe using the same frozen pair-state cache.
python -m scripts.train_frozen_idefics2_probe \
  data.artifact_path=/path/to/reranker_teacher_data.pkl \
  data.probe_cache_path=/path/to/frozen_idefics2_probe_cache \
  data.target=normalized_incremental_mean_token_probability \
  data.incremental_lambda=1.0

# Replay the selected pooled and frozen checkpoints on the same validation
# query/candidate ordering. The CSV contains one selection summary per query;
# the NPZ retains CLIP/pooled/frozen candidate ranks, both learned scores, and
# every stored Idefics2 outcome.
python -m scripts.export_reranker_predictions \
  --artifact /path/to/reranker_teacher_data.pkl \
  --pooled-checkpoint /path/to/pooled-run/best.pt \
  --frozen-checkpoint /path/to/frozen-run/best.pt \
  --probe-cache /path/to/frozen_idefics2_probe_cache \
  --output-dir /path/to/aligned-reranker-predictions

# K=32 closed-set controls remain available as true_probability and
# normalized_incremental_probability. They are useful comparators, but unlike
# the two targets above they normalize teacher scores over the audit label set.
```

Configurations live in [`configs/`](configs/). Run `pytest` for the GPU-free
unit suite.

Reranker experiment names default to `auto`. The generated directory name
contains architecture, target, objective, seed, and an eight-character hash of
the remaining data/model/objective/optimization configuration. Artifact paths
are excluded so copying the same artifact between machines preserves the run
identity. Set `experiment.name=<name>` only when a manual label is preferred.
