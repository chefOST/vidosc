"""MOSCATO annotations from chefOST-data, and source videos.

Alignment (verified for CMU): ground-truth state arrays have one entry per raw
video frame (30 fps), not the 10 fps timeline described in the MOSCATO README;
files_dict feature_len is a third of that (10 fps features).
"""
import json
import os
import subprocess
import urllib.request
from collections import Counter, defaultdict

DATASETS = {
    "CMU": {
        "gt": "annotations/ground_truth/CMU/gt_annotations_cmu.json",
        "pseudo": "annotations/pseudo_labels/CMU/pseudo_annotations_poc_new_cmu.json",
        "src_fps": 30,
    },
}
CMU_URL = "http://kitchen.cs.cmu.edu/Main/{}_Video.zip"


class Moscato:
    def __init__(self, root, dataset="CMU", cache_dir="data/cache"):
        if dataset not in DATASETS:
            raise NotImplementedError(f"{dataset}: only {list(DATASETS)} have verified frame alignment")
        self.root, self.dataset, self.cfg = root, dataset, DATASETS[dataset]
        self.src_fps = self.cfg["src_fps"]
        with open(os.path.join(root, self.cfg["gt"])) as f:
            self.gt = json.load(f)
        self.cache_dir = cache_dir

    @property
    def videos(self):
        return sorted(self.gt)

    def objects(self, vid):
        return sorted(self.gt[vid]["state"])

    def gt_states(self, vid, obj, frame_idx):
        seq = self.gt[vid]["state"][obj]
        return [set(seq[i]) if i < len(seq) else set() for i in frame_idx]

    def num_frames(self, vid):
        return max(len(s) for s in self.gt[vid]["state"].values())

    def candidate_states(self, source="pseudo_train"):
        """Object -> states it can take, from pseudo-labels only (no test-video information).
        'pseudo_train': states seen with the object in pseudo-labels of videos outside the GT split.
        'pseudo_recipe': same, restricted to training videos of the same recipe; returns
                         {recipe: {object: states}}, falling back to 'pseudo_train' per object.
        'all': every state in the pseudo-labels."""
        path = os.path.join(self.cache_dir, f"candidates_{self.dataset}_{source}.json")
        if os.path.exists(path):
            return json.load(open(path))
        counts = self.state_counts()
        every = sorted({s for c in counts.values() for s in c})
        by_obj = defaultdict(set)
        for (_, obj), c in counts.items():
            by_obj[obj].update(c)
        if source == "all":
            out = {o: every for o in by_obj}
        elif source == "pseudo_recipe":
            out = defaultdict(dict)
            for (rec, obj), c in counts.items():
                out[rec][obj] = sorted(c)
        else:
            out = {o: sorted(s) for o, s in by_obj.items()}
        os.makedirs(self.cache_dir, exist_ok=True)
        json.dump(out, open(path, "w"), indent=1)
        return out

    def candidates_for(self, cands, source, vid, obj):
        """The candidate list for one (video, object) under `source`."""
        if source != "pseudo_recipe":
            return cands.get(obj)
        own = cands.get(recipe(vid), {}).get(obj)
        if own:
            return own
        pooled = sorted({s for objs in cands.values() for s in objs.get(obj, [])})
        return pooled or None

    def state_counts(self):
        """(recipe, object) -> Counter of state -> frames, over pseudo-labels of training videos only."""
        if hasattr(self, "_counts"):
            return self._counts
        with open(os.path.join(self.root, self.cfg["pseudo"])) as f:
            pseudo = json.load(f)
        counts = defaultdict(Counter)
        for vid, ann in pseudo.items():
            if vid in self.gt:
                continue
            for obj, seq in ann["state"].items():
                for states in seq:
                    counts[(recipe(vid), obj)].update(set(states))
        self._counts = counts
        return counts

    def state_prior(self, vid, obj):
        """State -> pseudo-label frame count for this object in training videos of the video's recipe
        (all recipes if the object is unseen in this one)."""
        counts = self.state_counts()
        c = counts.get((recipe(vid), obj))
        if not c:
            c = Counter()
            for (_, o), cc in counts.items():
                if o == obj:
                    c.update(cc)
        return c


def recipe(vid):
    return vid.split("_")[1].lower()


def cmu_video(vid, video_dir="data/cmu_videos"):
    """Download (once) the CMU-MMAC zip for a MOSCATO video id and extract its camera file."""
    os.makedirs(video_dir, exist_ok=True)
    avi = os.path.join(video_dir, vid + ".avi")
    if not os.path.exists(avi):
        zip_path = os.path.join(video_dir, "_".join(vid.split("_")[:2]) + "_Video.zip")
        if not os.path.exists(zip_path):
            print(f"[video] downloading {os.path.basename(zip_path)}")
            urllib.request.urlretrieve(CMU_URL.format("_".join(vid.split("_")[:2])), zip_path)
        subprocess.run(["unzip", "-o", "-q", zip_path, vid + ".avi", "-d", video_dir], check=True)
    return avi


def sample_frames(video, src_fps, fps, frame_dir):
    """Every (src_fps / fps)-th raw frame as JPEG (cached). Returns (raw frame indices, image paths)."""
    step = round(src_fps / fps)
    os.makedirs(frame_dir, exist_ok=True)
    if not any(f.endswith(".jpg") for f in os.listdir(frame_dir)):
        subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"select=not(mod(n\\,{step})),scale=-2:336",
                        "-vsync", "vfr", "-q:v", "2", os.path.join(frame_dir, "%06d.jpg")], check=True)
    paths = sorted(os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    return [k * step for k in range(len(paths))], paths
