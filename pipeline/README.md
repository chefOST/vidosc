# Few-video pipeline

Tracks the states of given objects in a few MOSCATO videos and scores them with the README precision metric. ChefOST experiments, not paper results.

Stages, each swappable by flag: frame sampling → `--encoder` → state candidates (`--state_source`) and prompts (`--templates`) → `--decoder` → metric.

```bash
python -m pipeline.run --videos S12_Sandwich_7150991-2470 S48_Salad_7150991-222 \
    --templates basic --decoder viterbi:2 --wandb
```

The first approach is zero-shot: VidOSC's VLM pseudo-labeling step, generalized from initial/end to the object's MOSCATO state set. It outputs one state per sampled frame.

## Data

- Annotations: sibling clone of [chefOST-data](https://github.com/chefOST/chefOST-data) (`--data_root`).
- Videos: CMU-MMAC zips download on first use into `data/`. Only CMU is supported until EGTEA/EgoPER alignment is verified.
- Alignment (CMU): ground-truth arrays have one entry per raw 30 fps frame, not the 10 fps timeline described in the MOSCATO README. Sampling at 1 fps scores every 30th entry.

## Metric

Evaluated frames: sampled frames (1 fps) where the object has ≥1 ground-truth state. Precision: the predicted state is in the ground-truth set. Diagnostics, not metrics: `oracle` (some GT state is among the candidates) and `chance` (random pick from the candidates). Decoders use past and future frames, so this is offline, not causal tracking. Detail is not computed.

Outputs go to `runs/<name>/` (`results.json`, timeline plots). `--wandb` logs to the experiments project in [settings.json](../weights_and_biases/settings.json).

## Attribute pipeline (proposal)

`run_attributes.py` tracks each object as attribute values per frame (form, texture, heat, fill, surface, topping, location, use) instead of one state, and scores each attribute: correct if the predicted value is among those the frame's GT states assert. The mapping is `state_attributes/CMU.json` (draft, needs review). Not the README metric.

```bash
python -m pipeline.run_attributes --videos S08_Eggs_7150991-727 --values all --adapt_teacher state_zs
```

Methods: `prior`, `state_zs` (state zero-shot summed into attributes), `zeroshot` (attribute-phrase prompts), `adapt` (VidOSC trunk + attribute heads shared across objects, conditioned on the object's text embedding). `nontrivial_accuracy` drops (object, attribute) pairs with one candidate value. MOSCATO only asserts states, so attributes with a single asserted value (e.g. `use`) cannot be scored meaningfully.

## Embedding pipeline (proposal)

`run_embed.py`: methods output a vector per frame and object in the encoder's text space, not a state name. Scoring embeds the object's candidate states with the same text encoder and ranks them: `top1` (nearest candidate is a GT state) and `mrr`. Methods: `zeroshot` (the frame embedding), `adapt` (VidOSC trunk + projection into text space, self-trained without MOSCATO labels or states; `--adapt_target frame|changeit`). Not the README metric.

```bash
python -m pipeline.run_embed --videos S08_Eggs_7150991-727 --smooth 5
```
