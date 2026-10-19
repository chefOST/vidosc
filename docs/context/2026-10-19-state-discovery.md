Continue the embeddings work (state discovery, below). Two tasks:
  1. Confirm the results on more videos. Run all GPU stages on Modal (`pipeline/discovery/modal_jobs.py`);
     let me know before a particularly costly run.
  2. Investigate whether we can use MATCHA (https://aclanthology.org/2026.findings-acl.1055.pdf,
     code https://github.com/Siran-Li/MATCHA): a text-matching metric trained to separate
     contradictory texts that embedding similarity scores as near-identical.
  Keep the ground rules: zero-shot w.r.t. MOSCATO (GT, pseudo-labels and state lists only for
  scoring). Training on other data is allowed. Log milestones on W&B and update
  `writeup/state_discovery/state_discovery.tex` if results change.

## Context (as of 2026-10-19)

Write-up: `writeup/state_discovery/state_discovery.tex` (+ PDF). W&B: [ryan/discover-summary](https://wandb.ai/chefOST/vidosc-experiments/runs/sd6nfk81) (final tables, paired CIs), [ryan/discover-test-crops](https://wandb.ai/chefOST/vidosc-experiments/runs/ujbyh7zf). Earlier handoff: [ChefOST embedding-output eval: handoff](https://claude.ai/code/artifact/5224619e-81b2-45e7-adb0-7e11bb9b1ad7).

Pipeline (input: video + object name):
1. Track: SigLIP SO400M embedding per second, of the whole frame or of an OWLv2 crop (detector prompted with the object name; its score is a visibility weight).
2. Discovery: visibility-weighted smoothing, exact DP change points (penalty x noise per segment, min 5 s), merging of similar segments. The number of states comes from the data, and states can recur.
3. Naming (optional): Qwen2.5-VL-3B describes each state from 2 keyframes (about 7 calls per track); descriptions are embedded with SigLIP's text tower. `fused` = image centroid (minus the mean image embedding) + description embedding.

Scoring (MOSCATO only here): rank the object's candidate states (`pseudo_train`, embedded as text) by cosine to each frame's output: top-1, MRR, P@2/R@2, exact chance. Discovered segments vs GT state-set segments: MoF, NMI/ARI on tracks whose GT changes, change F1 within 5 s, number of states. Settings were chosen on dev (2 GT videos per recipe; 10 videos), and results are reported on the other 20 GT videos.

Test results:
- Fused (whole frame): top-1 .438 / MRR .580 vs per-frame SigLIP .353 / .521. That is +0.086 top-1 (CI +0.046 to +0.123); chance .171 / .361.
- Fused also beats discovery alone (+0.069) and the VLM alone (+0.064). Keyframes picked per state are not significantly better than evenly spaced ones.
- Discovery (crops): 3.0 states per track vs GT 2.9; change F1 .331 vs .247 for evenly spaced segments. One constant state still wins on MoF (.735 vs .729).

Caveats:
- Dev shows no significant fused gain (+0.035, CI −0.015 to +0.086), and crops beat whole frames on dev but not on test.
- The whole-frame fused run was added after seeing the crop results on test.
- The pipeline and the scorer share SigLIP's text tower.
- Whole frames can follow the scene rather than the object (bowl described as a pan).
- VidOSC is not used in the final pipeline. Earlier VidOSC adaptation scored below zero-shot.

How MATCHA might help (to investigate):
- Scorer: text-to-text state similarity is bunched (e.g. raw vs not raw 0.91). MATCHA could match VLM descriptions to GT states and penalise contradictions. It would also be independent of the pipeline's encoder.
- Pipeline: replace the cosine between description embeddings and states, or use its contrastive training recipe for state text.
- Check: does it handle short state phrases, licence and compute, and does it agree with an LLM judge on a sample?

More videos: only CMU has verified frame alignment (`pipeline/data.py`). EgoPER/EGTEA need alignment checks first. For CMU, MOSCATO non-GT videos have no GT, so more labelled test data means new datasets.

Code: `pipeline/discovery/{objects,discover,describe,modal_jobs,run,compare}.py`, plus top-k and chance-MRR scoring in `pipeline/embedding/embed.py`. Results: `runs/disc/{dev,test}-*-final`.

Modal: app `chefost-vidosc`, volume `chefost-vidosc` (frames under `/frames`), HF cache volume `chefost-hf-cache`. A10G runs cost a few dollars in total so far.

```bash
.venv/bin/modal run -m pipeline.discovery.modal_jobs::detect_all
.venv/bin/modal run -m pipeline.discovery.modal_jobs::crops_all
python -m pipeline.discovery.run --feats siglip --debias 1 --split test --penalty 500 --merge 0.05   # VLM calls go to Modal
python -m pipeline.discovery.compare disc/test-siglip-vlm-final:fused disc/test-siglip-final:frame
```
