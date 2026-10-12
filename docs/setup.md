# Setup

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
SP=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
echo "$PWD/third_party/lookforthechange_cpu" > "$SP/lookforthechange_cpu.pth"  # CPU port of VidOSC's CUDA op
echo "$PWD/third_party/compat" > "$SP/vidosc_compat.pth"                     # torchtext vocab shim
```

- VidOSC (`VidOSC/`): vendored from [facebookresearch/VidOSC](https://github.com/facebookresearch/VidOSC) at 5957577 (CC-BY-NC), with ChefOST changes to `train_changeit.py` and the added `eval_changeit_all.py`. ChangeIt data and checkpoints: see `VidOSC/README.md`. Run ChangeIt evals with `--accelerator cpu`.
- Pipeline (`pipeline/`): needs a sibling clone of [chefOST-data](https://github.com/chefOST/chefOST-data) (Git LFS). For `--encoder internvideo`:

```bash
git clone --filter=blob:none --sparse https://github.com/OpenGVLab/InternVideo.git third_party/InternVideo
git -C third_party/InternVideo checkout 3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3
git -C third_party/InternVideo sparse-checkout set InternVideo1/Pretrain/Multi-Modalities-Pretraining
.venv/bin/hf auth login   # after accepting the terms at huggingface.co/OpenGVLab/InternVideo1.0
.venv/bin/hf download OpenGVLab/InternVideo1.0 InternVideo-MM-L-14.ckpt --local-dir third_party/weights
```
