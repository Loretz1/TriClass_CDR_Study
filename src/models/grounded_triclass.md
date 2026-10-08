# Static Tri-Class Pseudo Supervision

This implementation adds new model/config files without changing the benchmark,
existing backbones, legacy caches, inference scoring, or API configuration.
It implements the overlap-supervised classifier, NOT the proposed target-only
simulated-query classifier. Dynamic expansion and residual scoring are disabled.

## Data Protocol

- Reuse completed source/overlap interest and target-item semantic embeddings.
- Train target LightGCN and user-level G only on target-only training users.
  All overlap target edges are excluded from these teachers. This deliberately
  differs from the old G cache so an overlap query cannot leak its own labels.
- Reuse a fingerprint-checked source-only LightGCN when available.
- Divide overlap users into fixed 60/20/20 train/validation/test groups.
- Merge up to 50 candidates each from semantic retrieval, G, and popularity.
  Popularity also uses target-only training edges. Deduplicate candidate IDs.
- Four inputs: semantic cosine, G cosine, exact-item interaction votes from
  source-similar overlap TRAIN users (excluding self), and log(1 + popularity).
- Labels: Hit takes precedence; Near is in the Top-20 item-neighbor union of
  the user's hidden target training positives; otherwise Unsupported. Near
  uses the isolated target teacher. Untrained items do not seed neighbors.
  Unsupported is unobserved support, NOT an explicit dislike.
- Train a 4-to-3 linear softmax classifier using natural-prior cross-entropy.
  Fit standardization on training users only. Early-stop on validation CE.
- Choose thresholds on validation users only, maximizing Hit precision plus
  eta times Near fraction subject to 20% user coverage. This is a proxy utility,
  not an estimate of actual recommendation improvement. Test users only audit.
- Never read valid_cold_tgt.pkl or test_cold_tgt.pkl in the offline builder.

## Training Contract

All four backbones consume the same cold_support.npz and class probabilities.
For eta in [0, 0.25, 0.5], weight = p_hit + eta * p_near. Keep at most five
candidates above the validation-selected threshold. Do not fill empty slots.
Sample 16 cold users uniformly per mini-batch, with five slots each. The loss
is sum(weight * BPR) / (16 * 5), not divided by the sum of weights. Empty slots
have zero weight. Negatives exclude the FULL candidate pool and any known
training positives resolved through raw user IDs, not coincident domain IDs.

Backbone settings match the current Grounded_Backbone configs; CD-CDR uses
CD_CDR_Normal. Only pseudo_loss_weight and near_support_weight are searched:
[0.01, 0.1, 1.0] x [0, 0.25, 0.5], seed 999, nine combinations per direction.
The existing benchmark launcher/evaluation logic is not changed. Its historical
Best Parameters summary uses test metrics; do not treat that summary as a
validation-selected result when reporting the new search.

## Commands

Run from src with the existing benchmark Python environment:

```bash
python -m models.grounded_triclass_pseudo --dataset Amazon2014 --domains Clothing_Shoes_and_Jewelry Sports_and_Outdoors
python -m models.grounded_triclass_pseudo --dataset Amazon2014 --domains Cell_Phones_and_Accessories Electronics
python main.py --model EMCDR_Grounded_TriClass_Pseudo --dataset Amazon2014 --domains Clothing_Shoes_and_Jewelry Sports_and_Outdoors
```

Other model names are UniCDR_Grounded_TriClass_Pseudo,
CDRIB_Grounded_TriClass_Pseudo, and CD_CDR_Normal_Grounded_TriClass_Pseudo.
Use --until prepare, teachers, or candidates to stop the offline builder early.
Use --config PATH to select a distinct output_subdir for changed settings.
No network requests or API calls are made by this builder.

## Outputs And Resume

Artifacts live under the existing dataset split's grounded_triclass directory:
build_spec.json, overlap_user_split.npz, target_only_teachers/, source_teacher/
(only if needed), overlap_candidates.npz, cold_candidates.npz,
target_item_neighbors.npy, calibrator.pt, overlap_support.npz, cold_support.npz,
diagnostics.json, manifest.json, and build.log. Re-running reuses completed
stages. Input hashes prevent mixing changed settings, code, data, or semantics.
A process lock prevents concurrent builders writing to the same output.
Candidate/probability caches use atomic writes; the final manifest is written
last. Synthetic test results are not experimental measurements.

## Verification

```bash
PYTHONPATH=. python -m unittest discover -s tools -p test_grounded_triclass.py -v
```

Tests cover label priority, user isolation, self-exclusion, probability fitting,
validation-only selection, slot normalization, abstention, negative sampling,
domain-local ID collisions, deterministic sampling, and real backbone backward
passes for both profiles. CDRIB's existing implementation requires CUDA.
