# State-sensitive spaces, MATCHA-adjacent (ChefOST experiment, 2026-10-21/22)

Goal: a vector space on top of SigLIP SO400M where object state, not object identity or viewpoint, drives distance.
Training uses no MOSCATO data; MOSCATO GT only scores. W&B: [ryan/state-space-matcha-adjacent](https://wandb.ai/chefOST/vidosc-experiments/runs/1xtnrlfy).
Background: [MATCHA](https://aclanthology.org/2026.findings-acl.1055.pdf) = small projector on frozen embeddings, margin loss on (reference, paraphrase, counterfactual contradiction). Earlier: [`experiments/matcha`](../matcha/README.md).

## Diagnosis (where SigLIP is state-blind)
- Text is fine: paraphrase vs LLM counterfactual, held-out EPIC nouns: .936.
- Image–text is not: EPIC frame at an action's start/end vs its before/after phrase (held-out participants): .538; state-changing verbs only: .487 (chance .5).
- CMU tracks: windowed embedding change does not peak at GT state changes (change AUC .52 frames, .47 crops). A GT time-scale/offset sweep found no better alignment, so this is viewpoint/motion dominating, not misalignment.

## What was tried
| approach | file | result |
|---|---|---|
| LDA state subspace (within-object state vs object-identity scatter, EPIC text) | `space.py` | less bunching (.67→.55) but description→state test top-1 drops (.404→.30–.37) |
| MLP head, MATCHA text triplets (5k LLM paraphrase/counterfactual sets, `gen_triplets.py`) + frame margins | `train.py` | EPIC 2AFC .54→.57, but CMU dev per-frame top-1 .296→.179 (breaks image–text alignment) |
| + cross-modal InfoNCE, keep-distillation, state-verb/no-location filters | `train.py` | no held-out gain (≤.542); noun retrieval falls .24→.06–.20 |
| image-only triplets from EPIC action structure (untouched gap = same state, across a state verb = other state) | `train_gap.py` | whole frames: memorises 360–1.5k sources; held-out ≤ SigLIP (.70). Noun crops (64 videos, `modal_crops.py`): SigLIP .60/.575, heads .55–.61 (within noise) |
| contradiction-axis space: blind LLM writes each object's contradictory state pairs from its name only; project onto e(o,s+)−e(o,s−) | `axes.py` | CMU dev: change AUC .523→.576 [+.016,+.094], segment separability +.11; naming drops (−.08) |

Takeaway: with a few thousand EPIC frames and LLM-guessed phrases, trained heads overfit or break alignment (as the earlier distiller/LoRA did). The training-free contradiction axes are the only variant that increases state sensitivity of tracks.

## Contradiction axes for discovery only (`run.py --seg_space axes-proj`)
States discovered on the axis track; all outputs (centroids, keyframes, VLM) stay in SigLIP. Penalty 800 matches SigLIP's states per track (label-free, but picked after seeing these splits). Paired video bootstrap, `compare_disc.py`:

| | ARI | NMI | change F1 | MoF | discover top-1 |
|---|---|---|---|---|---|
| test, frames | +.037 [+.003,+.073] | +.035 [−.001,+.070] | −.019 [−.058,+.016] | +.031 [+.003,+.059] | +.024 [−.005,+.050] |
| test, crops | +.030 [−.004,+.066] | +.031 [−.001,+.067] | +.002 | +.009 | +.003 |
| dev, frames | +.014 | +.011 | −.042 [−.085,+.013] | −.012 | +.011 |
| dev, crops | +.015 | +.002 | −.038 | +.008 | −.013 |

Better grouping on test, nothing on dev, boundaries never better. Suggestive, not established. Not run: `fused` with axis segments (needs new VLM keyframes; Modal) and EGTEA.

## Data and code
- `corpus.py` EPIC text/frames; `data/epic/state_triplets.json` (5k phrases, blind Opus); `data/cache/statespace/axes.json` (80 CMU+EGTEA objects, blind, name only).
- New unlabeled data: non-GT CMU frame embeddings (40 videos, `data/cmu_nongt.txt`), EPIC frame embeddings 39→64 videos, EPIC noun crops 64/64 (`modal_crops.py`; Modal failures came from unbounded parallel A10Gs, fixed with `max_containers=6`). Not yet used in a successful model.
- `pipeline/discovery/run.py`: `--space`, `--seg_space`, per-pair discovery scores; text cache now stored as keys + matrix (68k prompts load in 0.1 s instead of ~17 min).
- Evals: `eval_desc.py` (description→state, same items as MATCHA), `eval_tracks.py` (change AUC, separability, frame/segment ranking).

## Pre-registered run (written 2026-10-22, before running)
Whole frames, Qwen2.5-VL-3B, all methods. Primary metric: `fused` top-1, paired video bootstrap vs the same config without `--seg_space`.
- CMU test: `--seg_space axes-proj --penalty 800` vs `--penalty 500` (write-up setting).
- EGTEA (all 12): `--seg_space axes-proj --penalty 800 --len_ref 268` vs `--penalty 500 --len_ref 268`.
Run once each; no further settings on these splits.

Result: negative. Baselines reproduce the write-up (CMU `fused` .438, EGTEA .294).
| | `fused` top-1 | ARI | change F1 | states/track |
|---|---|---|---|---|
| CMU test | .448 vs .438, +.010 [−.018,+.038] | +.037 [+.003,+.073] | −.019 n.s. | 3.40 vs 3.67 |
| EGTEA | .280 vs .294, −.014 [−.060,+.028] | −.024 n.s. | −.082 [−.131,−.035] | 2.29 vs 2.81 (GT 3.8) |

Axis segmentation groups CMU frames slightly better but does not improve naming, and on EGTEA it under-splits and
misplaces changes. Not adopted. W&B [ryan/state-space-preregistered](https://wandb.ai/chefOST/vidosc-experiments/runs/acwke4ax).

## Text matchers for description → state (2026-10-24, `modal_textmatch.py`, `score_textmatch.py`)
Same items as `eval_desc.py` (Qwen description vs the object's candidate states). Top-1, paired video bootstrap vs centred SigLIP text. W&B [ryan/state-text-matchers](https://wandb.ai/chefOST/vidosc-experiments/runs/lap1go7z).

| matcher | dev | test |
|---|---|---|
| SigLIP text, centred | .322 | .404 |
| Qwen3-Embedding-4B, no instruction | .302 | .424 (+.020 [−.005,+.044]) |
| Qwen3-Embedding-4B / 8B, instruction | .298 / .312 | .382 / .381 |
| bge-large (no instr. / instr.) | .308 / .282 | .396 / .389 |
| Qwen3-Reranker-4B | .277 | .365 (−.039 [−.059,−.017]) |
| Qwen3-8B yes/no judge | .261 | .268 |
| NLI DeBERTa-v3-large | .223 | .263 |

Nothing beats SigLIP on dev. On dev, 48% of items are missed by every matcher (random .22). These descriptions don't support the GT states ("bowl is full of chocolate" → GT covered/dirty/empty/partially-empty). The bottleneck is the description, not the text space.

## Intrinsic separation test (2026-10-24, `eval_sep.py`, `modal_textmatch.py::sep`)
Short write-up of the separation test: [`writeup/state_space`](../../writeup/state_space/state_space.pdf).
No VLM in the loop. Each item is an anchor state with 3 paraphrases (new wording) and 4 contradictions (kept wording, one aspect flipped), all from a blind LLM. Pair acc = P[s(anchor, para) > s(anchor, contra)]; strict = every paraphrase beats every contradiction. Bootstrap over objects. W&B [ryan/state-separation-intrinsic](https://wandb.ai/chefOST/vidosc-experiments/runs/ihjd49ii).

| space | EPIC phrases (5k) acc / strict | MOSCATO state names (357, eval only) acc / strict |
|---|---|---|
| SigLIP centred | **.930** / .655 | .843 / .401 |
| Qwen3-Embedding-8B | .899 / .575 | **.915** / .602 (+.072 [+.053,+.091]) |
| Qwen3-Embedding-4B | .894 / .559 | .906 / .543 |
| bge-large | .869 / .488 | .913 / .599 |
| LDA k=256 | .873 / .498 | .830 / .387 |
| contradiction axes (proj) | – | .720 / .347 |
| word overlap | .637 / .132 | .110 / .000 |

Rerun with 8 paraphrases and ~10 contradictions per state (≤2 per aspect; `-big` sets: all 357 MOSCATO names, 1000 random EPIC states). Strict is harsher at ~80 pairs per state, so compare it within a column only.

| space | EPIC-big acc / strict | MOSCATO-big acc / strict |
|---|---|---|
| SigLIP centred | **.941** / .408 | .869 / .132 |
| Qwen3-Embedding-8B | .923 / .347 | .913 / .241 (+.043 [+.029,+.058]) |
| Qwen3-Embedding-4B | .922 / .332 | .919 / .241 (+.049 [+.033,+.066]) |
| bge-large | .916 / .299 | **.927** / .300 (+.058 [+.041,+.078]) |
| LDA k=256 | .906 / .261 | .856 / .115 |
| contradiction axes (proj) | – | .739 / .090 |

The conclusion holds. The three text embedders are within noise of each other.

On terse MOSCATO-style names, where word overlap is fully adversarial, the text embedders beat SigLIP. The gap is largest on full/empty, wet/dry, clean/dirty and on/off (SigLIP .75–.79 vs .86–.94). On longer EPIC phrases SigLIP is best; its centring mean and the LDA were fit on EPIC states. Text-only, so this helps text↔text steps (description→state, state-name merging), not frame→state.

## Next
1. Axis track as a boundary-free merge signal only (keep SigLIP change points, merge in axis space), since grouping improves and timing does not.
2. Trained heads only with much more grounded data (EPIC crops for all videos; per-frame image-grounded targets).

## Food-only evaluation (2026-10-24)
The current writeup uses food subsets of the larger sets: 184 MOSCATO states (17 foods) and
367 EPIC phrases (249 food names). Foods, ingredients and drinks are included; utensils,
packaging, waste and ambiguous contents are excluded by an explicit reviewed allowlist.
Run `python experiments/statespace/eval_food.py` to rebuild the subsets and recompute metrics
and 2,000 paired bootstrap resamples using the original cached similarities.
Classification manifests: `data/cache/statespace/{moscato,epic}_food_classification.json`.
Results: `data/cache/statespace/food_sep_results.json`; food triplets are in the same directory.
Earlier mixed-set results above are retained for reference.

## Expanded food sets (2026-10-24)
`generate_food_additions.py` creates GPT-authored additions using explicit synonym and contradiction
substitutions. Selection covers additional food names first, then round-robin states, without preferred foods.
The expanded short-label set contains 416 states across 246 food names: 184 MOSCATO candidate pairs and
232 synthetic style-matched extensions. These extensions are not MOSCATO annotations.
The expanded descriptive set contains 600 EPIC-derived anchors across 480 food names (367 existing + 233 added).
Both sets retain 8 paraphrases and approximately 10 contradictions per anchor. Original entries were authored
by Claude; additions were authored by GPT-6 in Codex, with shared wording templates. Results are pooled within
each set; shared templates limit the independence of examples.

Reproduce:
```
python experiments/statespace/generate_food_additions.py
modal run experiments/statespace/modal_textmatch.py::foods_expanded
python experiments/statespace/eval_food.py --expanded
```
Data: `data/cache/statespace/{moscato,epic}_food_expanded_triplets.json`.
Scores and results: `food_expanded_scores.json`, `food_expanded_results.json` in the same directory.
The Claude-based `topup_food.py` is an alternative generator and was not used for these additions.

Expanded pooled results (pair accuracy / strict):
| Model | Short food labels (416 states) | Food phrases (600 states) |
|---|---|---|
| bge-large | .943 / .462 | .929 / .405 |
| Qwen3-Embedding-4B | .947 / .423 | .933 / .415 |
| Qwen3-Embedding-8B | .944 / .425 | .935 / .433 |
| SigLIP text | .936 / .457 | .956 / .530 |
| Word overlap | .387 / .053 | .735 / .082 |
Pair-accuracy differences among the four embedders are not significant on short labels.
SigLIP text exceeds all three general embedders on descriptive phrases (paired food-name bootstrap).
The published writeup is an experiment report; it covers these pooled synthetic evaluations.

## Two-way bootstrap (2026-10-24)
`eval_food.py --expanded` now uses 2,000 paired food-by-recorded-template pigeonhole resamples.
Food and template clusters are independently sampled with replacement. Each anchor receives the product
of the two cluster multiplicities; scores are normalized by the total replicate weight. All models share
replicate weights. Point scores and equal-anchor aggregation are unchanged.

Alternate surface forms from one GPT generation rule share a template family (67 recorded families).
The original 184/367 Claude examples lack template IDs and each gets a singleton template cluster;
unknown common patterns in those examples are not accounted for. Short labels have 246 food clusters
and 251 template clusters; phrases have 480 and 434. Original food-only intervals remain in the results
under `comparisons_food_only`; `comparisons` contains the primary two-way pair-accuracy intervals.
Strict-score comparisons are in `comparisons_strict_two_way`; assignments are in `bootstrap_groups`.

The conclusions are unchanged: no significant pair-accuracy difference among the four embedders on short
labels, and SigLIP text remains above the three general embedders on phrases. For example, its phrase
advantage over Qwen3-8B is +.0204, two-way 95% CI [+.0030,+.0372] (food-only [+.0122,+.0280]).
Run `python -m unittest test_food_bootstrap.py` from `experiments/statespace/` for the resampling checks.
