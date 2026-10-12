"""InternVideo-MM-L-14, the frozen encoder behind VidOSC's ChangeIt features.

Recipe matched against VidOSC's shipped ChangeIt features (median cosine 0.92 on a re-downloaded
video, `_232eHtLkBM`): per second, 8 frames spread evenly over a 1 s window, short side resized
to 224 (bilinear), centre-cropped to 224, CLIP normalization. Outputs are not L2-normalized,
as VidOSC expects. Here the window is centred on each sampled frame so it lines up with the
ground truth at that frame (ChangeIt features cover [t, t+1)).

Weights: huggingface.co/OpenGVLab/InternVideo1.0 (gated), InternVideo-MM-L-14.ckpt.
"""
import os
import subprocess
import sys
import types

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "third_party", "InternVideo", "InternVideo1", "Pretrain", "Multi-Modalities-Pretraining")
WEIGHTS = os.path.join(ROOT, "third_party", "weights", "InternVideo-MM-L-14.ckpt")
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def _import_internvideo():
    # The package imports decord (video loading we don't use) and pkg_resources at import time.
    import packaging.version
    sys.modules.setdefault("decord", types.ModuleType("decord"))
    if "pkg_resources" not in sys.modules:
        pr = types.ModuleType("pkg_resources")
        pr.packaging = packaging
        sys.modules["pkg_resources"] = pr
    sys.path.insert(0, CODE)
    import InternVideo
    return InternVideo


def _prep(rgb):
    im = Image.fromarray(rgb)
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    x = torch.from_numpy(np.array(im.crop((l, t, l + 224, t + 224)))).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


def clip_frames(center, src_fps, n_frames):
    """Raw frame indices of the 8-frame, 1 s clip centred on raw frame `center`."""
    return [min(max(round(center + (j - 3.5) / 8 * src_fps), 0), n_frames - 1) for j in range(8)]


def video_size(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,nb_read_packets", "-of", "csv=p=0", video],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1]), int(out[2])


class InternVideoEncoder:
    """'internvideo' or 'internvideo:<ckpt path>'. Video-level: encode_video() takes the raw video."""
    video_level = True

    def __init__(self, ckpt=WEIGHTS, device=None):
        iv = _import_internvideo()
        self.tokenize = iv.tokenize
        self.name = "internvideo-mm-l14"
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = iv.load_model(ckpt).float().eval().to(self.device)
        self.logit_scale = self.model.logit_scale.exp().item()

    @torch.no_grad()
    def encode_texts(self, texts):
        e = self.model.encode_text(self.tokenize(texts).to(self.device)).float()
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    @torch.no_grad()
    def encode_video(self, video, centers, src_fps, batch=8):
        """[len(centers), 768] raw (unnormalized) features, one 1 s clip per centre frame index."""
        w, h, n = video_size(video)
        clips = [clip_frames(c, src_fps, n) for c in centers]
        need = {i for c in clips for i in c}
        last = max(need)
        frames, out, done = {}, [], 0
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                stdout=subprocess.PIPE)
        for i in range(last + 1):
            buf = proc.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            if i in need:
                frames[i] = _prep(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
            # Encode every complete batch of clips as soon as its frames are decoded.
            while done < len(clips) and (len(clips) - done >= batch or i == last) and \
                    all(j in frames for c in clips[done:done + batch] for j in c):
                chunk = clips[done:done + batch]
                x = torch.stack([torch.stack([frames[j] for j in c], 1) for c in chunk]).to(self.device)
                out.append(self.model.encode_video(x).float().cpu())
                done += len(chunk)
                keep = min(clips[done]) if done < len(clips) else last + 1
                frames = {j: f for j, f in frames.items() if j >= keep}
        proc.kill()
        if done < len(clips):  # video shorter than ffprobe reported: reuse the last decoded frame
            lastf = frames[max(frames)]
            x = torch.stack([torch.stack([frames.get(j, lastf) for j in c], 1) for c in clips[done:]]).to(self.device)
            out.append(self.model.encode_video(x).float().cpu())
        return torch.cat(out).numpy()
