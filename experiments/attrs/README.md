# Attributes (ChefOST experiment, zero-shot)

Input: video, recipe name and object names. Output: one value per LLM-generated attribute per frame. Not published results. Writeup: [writeup/attrs](../../writeup/attrs/attrs.pdf).

1. `generate.py [n]`: blind LLM (fresh `claude -p` with no tools or context, via `llm.py`) → `schemas/schema_s{k}.json`. Each object gets attributes, values, captions, a script (expected order of values) and visibility captions.
2. Evidence. `modal_vlm.py` runs Qwen3-VL 8B/32B on Modal with vLLM, one object per prompt, a keyframe every 6 s → `vlm_cache/<schema>-<model>/`. `vlm_read.py` is the same reader run locally (MLX 4-bit; slow, partial run in `vlm_cache/s0-8b-mlx4bit-partial/`). The encoder baseline uses the cached features in `method.py`.
3. `method.py`: decoders (script soft/hard alignment, Viterbi, nearest keyframe).

Eval only: `evalmap.py` maps MOSCATO GT words to attribute values with an LLM that sees both (`schemas/evalmap_*`). `mapagree.py` measures agreement between two mappers. The metric (Acc, Acc_Δ, BAcc_Δ) is a proposal, not the README precision. See the writeup.

```
.venv/bin/modal run experiments/attrs/modal_vlm.py --schema s0 --model 32b    # from vidosc/
.venv/bin/python evaluate.py --schema s0 --vlm s0-32b [--map claude-sonnet-5-5] [--vlm_every 2] [--wandb --name ryan/...]
.venv/bin/python explore.py s0     # encoder/evidence variants (all below the script prior)
```

Results go to `runs/attrs-*/results.json`. W&B: group `attrs` in chefOST/vidosc-experiments (`ryan/attrs-m1-*`, `ryan/attrs-m2-*`).
