"""Stage 2b on Modal: same per-object keyframe prompts as vlm_read.py, served with vLLM (bf16 / FP8) on one GPU.
Usage (from vidosc/): .venv/bin/modal run experiments/attrs/modal_vlm.py --schema s1 --model 8b [--every 6] [--videos a,b]
-> experiments/attrs/vlm_cache/<schema>-<model>/<vid>.jsonl (same format as vlm_read.py)"""
import base64, json, os
import modal

MODELS = {'8b': ('Qwen/Qwen3-VL-8B-Instruct', 'A100-80GB'), '32b': ('Qwen/Qwen3-VL-32B-Instruct-FP8', 'H100')}
app = modal.App('chefost-attr-vlm')
image = modal.Image.debian_slim(python_version='3.12').pip_install('vllm==0.11.0', 'transformers==4.57.1', 'qwen-vl-utils==0.0.14')
hf = modal.Volume.from_name('chefost-hf-cache', create_if_missing=True)


class _Reader:
    model_id = None

    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(self.model_id, max_model_len=4096, limit_mm_per_prompt={'image': 1}, gpu_memory_utilization=.9)

    @modal.method()
    def run(self, reqs):
        """reqs: [(prompt text, jpeg bytes, max_tokens)] -> [text]"""
        from vllm import SamplingParams
        msgs = [[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(b).decode()}},
            {'type': 'text', 'text': p}]}] for p, b, _ in reqs]
        outs = self.llm.chat(msgs, [SamplingParams(temperature=0, max_tokens=m) for _, _, m in reqs], use_tqdm=False)
        return [o.outputs[0].text for o in outs]


kw = dict(image=image, volumes={'/root/.cache/huggingface': hf}, timeout=3600, scaledown_window=120, max_containers=1)


@app.cls(gpu='A100-80GB', **kw)
class ReaderA100(_Reader):
    model_id = MODELS['8b'][0]


@app.cls(gpu='H100', **kw)
class ReaderH100(_Reader):
    model_id = MODELS['32b'][0]


Readers = {'A100-80GB': ReaderA100, 'H100': ReaderH100}


@app.local_entrypoint()
def main(schema: str = 's1', model: str = '8b', every: int = 6, videos: str = ''):
    import sys, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vlm_read as R
    here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(os.path.dirname(here))
    gt = json.load(open(root + '/../chefOST-data/MOSCATO/annotations/ground_truth/CMU/gt_annotations_cmu.json'))
    objects = {v: sorted(gt[v]['state']) for v in gt}; del gt  # pipeline input: object names only
    sch = json.load(open(f'{here}/schemas/schema_{schema}.json'))
    mid, gpu = MODELS[model]; reader = Readers[gpu]()
    od = f'{here}/vlm_cache/{schema}-{model}'; os.makedirs(od, exist_ok=True)
    vids = videos.split(',') if videos else sorted(objects)
    t0 = time.time()
    for vi, vid in enumerate(vids):
        path = f'{od}/{vid}.jsonl'
        if os.path.exists(path): continue
        r = vid.split('_')[1]; groups = {}
        for it in R.items(sch, r, objects[vid]): groups.setdefault(it[0], []).append(it)
        fd = f'{root}/data/frames/{vid}_fps1'; frames = sorted(os.listdir(fd)); ts = list(range(0, len(frames), every))
        jpg = {t: open(f'{fd}/{frames[t]}', 'rb').read() for t in ts}
        reqs, keys = [], []
        for t in ts:
            for o, g in groups.items():
                reqs.append((R.prompt(r, o, g), jpg[t], 6 * len(g) + 10)); keys.append((t, o))
        outs = reader.run.remote(reqs)
        raw = {}
        for (t, o), txt in zip(keys, outs): raw.setdefault(t, {})[o] = txt
        with open(path + '.tmp', 'w') as f:
            for t in ts:
                ans = [x for o, g in groups.items() for x in R.parse(raw[t][o], g)]
                items_ = [list(x[:2]) for g in groups.values() for x in g]
                f.write(json.dumps({'t': t, 'raw': raw[t], 'items': items_, 'ans': ans, 'model': mid}) + '\n')
        os.replace(path + '.tmp', path)
        print(f'[{vi + 1}/{len(vids)}] {vid} calls={len(reqs)} elapsed={time.time() - t0:.0f}s', flush=True)
