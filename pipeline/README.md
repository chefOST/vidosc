# Few-video pipeline

Tracks the states of given objects in a few MOSCATO videos. ChefOST experiments, not paper results. Run everything from `vidosc/`.

## Layout

- Shared: `data.py` (MOSCATO, videos, frames), `encoders.py`/`internvideo.py`, `vidosc.py` (VidOSC wrappers), `scoring.py` (zero-shot state probabilities), `decoders.py`, `metrics.py` (README precision).
- [`states/`](#states): one state per frame, README metric.
- [`attributes/`](#attributes-proposal), [`embedding/`](#embedding-proposal), [`discovery/`](#state-discovery-proposal): proposals, not the README metric.

## Data

- Annotations: sibling clone of [chefOST-data](https://github.com/chefOST/chefOST-data) (`--data_root`).
- Videos: CMU-MMAC zips download on first use into `data/`. Only CMU is supported until EGTEA/EgoPER alignment is verified.
- Alignment (CMU): ground-truth arrays have one entry per raw 30 fps frame, not the 10 fps timeline described in the MOSCATO README. Sampling at 1 fps scores every 30th entry.

Outputs go to `runs/<name>/` (`results.json`, plots). `--wandb` logs to the experiments project in [settings.json](../weights_and_biases/settings.json).

## States

Stages, each swappable by flag: frame sampling → `--encoder` → state candidates (`--state_source`) and prompts (`--templates`) → `--decoder` → metric. Near-synonym clusters: `states/state_clusters/CMU.json` (`--clusters CMU`).

```bash
python -m pipeline.states.run --videos S12_Sandwich_7150991-2470 S48_Salad_7150991-222 \
    --templates basic --decoder viterbi:2 --wandb
```

The first approach is zero-shot: VidOSC's VLM pseudo-labeling step, generalized from initial/end to the object's MOSCATO state set. It outputs one state per sampled frame.

Metric. Evaluated frames: sampled frames (1 fps) where the object has ≥1 ground-truth state. Precision: the predicted state is in the ground-truth set. Diagnostics, not metrics: `oracle` (some GT state is among the candidates) and `chance` (random pick from the candidates). Decoders use past and future frames, so this is offline, not causal tracking. Detail is not computed.

## Attributes (proposal)

`attributes/run.py` tracks each object as attribute values per frame (form, texture, heat, fill, surface, topping, location, use) instead of one state, and scores each attribute: correct if the predicted value is among those the frame's GT states assert. The mapping is `attributes/state_attributes/CMU.json` (draft, needs review).

```bash
python -m pipeline.attributes.run --videos S08_Eggs_7150991-727 --values all --adapt_teacher state_zs
```

Methods: `prior`, `state_zs` (state zero-shot summed into attributes), `zeroshot` (attribute-phrase prompts), `adapt` (VidOSC trunk + attribute heads shared across objects, conditioned on the object's text embedding). `nontrivial_accuracy` drops (object, attribute) pairs with one candidate value. MOSCATO only asserts states, so attributes with a single asserted value (e.g. `use`) cannot be scored meaningfully. Superseded by [experiments/sliders](../experiments/sliders/README.md).

## Embedding (proposal)

`embedding/run.py`: methods output a vector per frame and object in the encoder's text space, not a state name. Scoring (`embedding/embed.py`) embeds the object's candidate states with the same text encoder and ranks them: `top1` (nearest candidate is a GT state), `mrr`, P@k/R@k. Methods: `zeroshot` (the frame embedding), `adapt` (VidOSC trunk + projection into text space, self-trained without MOSCATO labels or states; `--adapt_target frame|changeit`).

```bash
python -m pipeline.embedding.run --videos S08_Eggs_7150991-727 --smooth 5
```

## State discovery (proposal)

`discovery/run.py`: per (video, object), an embedding track (SigLIP frames or OWLv2 object crops, `objects.py`) is split into a data-chosen number of states (`discover.py`); states can be named by a VLM (`describe.py`). GPU stages run on Modal (`modal_jobs.py`, volume `chefost-vidosc`). Scoring ranks MOSCATO states (as in embedding) and compares discovered states to GT segments; `compare.py` gives paired bootstrap CIs. Dev/test split: 10/20 GT videos. Write-up: [writeup/state_discovery](../writeup/state_discovery/state_discovery.tex). Handoff: [docs/handoffs](../docs/handoffs/2026-10-19-state-discovery.md).

```bash
.venv/bin/modal run -m pipeline.discovery.modal_jobs::detect_all
python -m pipeline.discovery.run --feats siglip --debias 1 --split test --penalty 500 --merge 0.05
python -m pipeline.discovery.compare disc/test-siglip-vlm-final:fused disc/test-siglip-final:frame
```
