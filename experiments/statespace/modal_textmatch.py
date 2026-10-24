"""Text matchers for description -> candidate state, on the eval_desc items (runs/matcha/items.json; MOSCATO GT
used only to score). Pick on dev, report test once.

  bi-encoders  Qwen3-Embedding-4B/8B, bge-large-en-v1.5 (with / without the query instruction)
  reranker     Qwen3-Reranker-4B (pointwise yes/no)
  nli          cross-encoder/nli-deberta-v3-large (entailment - contradiction logit)
  judge        Qwen3-8B, pointwise yes/no logit, thinking off

    modal run experiments/statespace/modal_textmatch.py   # -> data/cache/statespace/textmatch.json, prints table
"""
import json
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "data", "cache", "statespace", "textmatch.json")
hf = modal.Volume.from_name("chefost-hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==4.57.1", "accelerate", "sentence-transformers==5.1.0", "numpy",
                      "open_clip_torch==3.3.0", "timm", "pillow")
         .env({"HF_HOME": "/hf"})
         .add_local_python_source("pipeline"))
app = modal.App("chefost-statespace", image=image)

TASK = "Given a description of an object seen in a cooking video, retrieve the object's current physical state"
BI = {  # name: (model, query prompt or None)
    "qwen3-emb-4b": ("Qwen/Qwen3-Embedding-4B", f"Instruct: {TASK}\nQuery: "),
    "qwen3-emb-4b-noinst": ("Qwen/Qwen3-Embedding-4B", None),
    "qwen3-emb-8b": ("Qwen/Qwen3-Embedding-8B", f"Instruct: {TASK}\nQuery: "),
    "bge-large": ("BAAI/bge-large-en-v1.5", "Represent this sentence for searching relevant passages: "),
    "bge-large-noinst": ("BAAI/bge-large-en-v1.5", None),
}


def query(obj, desc):
    return f"{obj}: {desc}"


def doc(obj, state):
    return f"{obj}: {state}"


@app.function(gpu="L40S", volumes={"/hf": hf}, timeout=3600)
def bi_encoder(name, pairs):
    """pairs: [(obj, desc, state)] -> cosine per pair."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    model_id, prompt = BI[name]
    m = SentenceTransformer(model_id, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    qs = sorted({query(o, d) for o, d, _ in pairs})
    ds = sorted({doc(o, s) for o, _, s in pairs})
    Q = m.encode(qs, prompt=prompt, batch_size=64, normalize_embeddings=True)
    D = m.encode(ds, batch_size=64, normalize_embeddings=True)
    qi, di = {q: k for k, q in enumerate(qs)}, {d: k for k, d in enumerate(ds)}
    return name, [float(np.dot(Q[qi[query(o, d)]], D[di[doc(o, s)]])) for o, d, s in pairs]


def _yes_no(model_id, texts, bs=64):
    """log p(yes) - log p(no) at the last position of each (already templated) prompt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            x = tok(texts[i:i + bs], padding=True, return_tensors="pt").to("cuda")
            lg = m(**x).logits[:, -1, :].float()
            out += (lg[:, yes] - lg[:, no]).tolist()
    return out


@app.function(gpu="L40S", volumes={"/hf": hf}, timeout=3600)
def reranker(pairs):
    pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
           "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    texts = [f"{pre}<Instruct>: {TASK}\n<Query>: {query(o, d)}\n<Document>: {doc(o, s)}{suf}" for o, d, s in pairs]
    return "qwen3-rerank-4b", _yes_no("Qwen/Qwen3-Reranker-4B", texts)


@app.function(gpu="L40S", volumes={"/hf": hf}, timeout=3600)
def judge(pairs):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    q = ("A frame from a cooking video shows a {o}. A captioner described it as: \"{d}\".\n"
         "Is the {o}'s current state \"{s}\"? Answer only yes or no.")
    texts = [tok.apply_chat_template([{"role": "user", "content": q.format(o=o, d=d, s=s)}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for o, d, s in pairs]
    return "qwen3-8b-judge", _yes_no("Qwen/Qwen3-8B", texts)


@app.function(gpu="L40S", volumes={"/hf": hf}, timeout=3600)
def nli(pairs):
    from sentence_transformers import CrossEncoder
    m = CrossEncoder("cross-encoder/nli-deberta-v3-large", device="cuda")
    lg = m.predict([(f"The {o}: {d}.", f"The {o} is {s}.") for o, d, s in pairs], batch_size=128)
    return "nli-deberta-v3-large", [float(r[1] - r[0]) for r in lg]  # labels: contradiction, entailment, neutral


@app.local_entrypoint()
def main():
    sys.path.insert(0, HERE)
    from eval_desc import ITEMS
    items = json.load(open(ITEMS))
    pairs = sorted({(i["obj"], i["desc"], s) for i in items for s in i["states"]})
    print(f"{len(items)} items, {len(pairs)} unique pairs")
    calls = [bi_encoder.spawn(n, pairs) for n in BI] + [f.spawn(pairs) for f in (reranker, judge, nli)]
    scores = {}
    for c in calls:
        try:
            name, s = c.get()
            scores[name] = s
            print("done", name)
        except Exception as e:  # keep the other matchers
            print("failed", repr(e))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"pairs": pairs, "scores": scores}, open(OUT, "w"))
    print("wrote", OUT)


# --- intrinsic separation test (eval_sep.py): same-state vs contradicting phrase pairs, vector models only ---
SEP_BI = {"qwen3-emb-4b": "Qwen/Qwen3-Embedding-4B", "qwen3-emb-8b": "Qwen/Qwen3-Embedding-8B",
          "bge-large": "BAAI/bge-large-en-v1.5"}


@app.function(gpu="L40S", volumes={"/hf": hf}, timeout=3600)
def sep_bi(name, triples):
    """triples: [(obj, a, b)] -> cosine of '{obj}: a' and '{obj}: b' (symmetric, no instruction)."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(SEP_BI[name], device="cuda", model_kwargs={"torch_dtype": torch.float16})
    ts = sorted({f"{o}: {x}" for o, a, b in triples for x in (a, b)})
    E = m.encode(ts, batch_size=128, normalize_embeddings=True)
    ix = {t: k for k, t in enumerate(ts)}
    return name, [float(np.dot(E[ix[f"{o}: {a}"]], E[ix[f"{o}: {b}"]])) for o, a, b in triples]


@app.function(gpu="L40S", volumes={"/hf": hf}, timeout=3600)
def siglip_texts(texts):
    """unit SigLIP SO400M text embeddings (fp16 bytes), same encoder as the pipeline's TextBank."""
    import numpy as np
    from pipeline import encoders
    enc = encoders.CLIPEncoder("ViT-SO400M-14-SigLIP-384/webli", device="cuda")
    E = np.concatenate([enc.encode_texts(texts[i:i + 512]) for i in range(0, len(texts), 512)])
    return E.astype(np.float16).tobytes()


@app.local_entrypoint()
def sep():
    import numpy as np
    sys.path.insert(0, HERE)
    import eval_sep
    from corpus import prompt
    from space import Space, fit_lda, text_mean
    from axes import AxisSpace
    sets = {n: eval_sep.pairs(eval_sep.load(n)) for n in eval_sep.SETS}
    texts = sorted({prompt(o, x) for ps in sets.values() for o, a, b in ps for x in (a, b)})
    print({n: len(p) for n, p in sets.items()}, len(texts), "siglip prompts")
    calls = {n: [sep_bi.spawn(m, p) for m in SEP_BI] for n, p in sets.items()}
    E = np.frombuffer(siglip_texts.remote(texts), np.float16).reshape(len(texts), -1).astype(np.float32)
    ix = {t: k for k, t in enumerate(texts)}
    mu = text_mean()
    spaces = {"siglip-raw": None, "siglip-centred": Space(mu), "lda-256": fit_lda(256)[0]}
    axes = AxisSpace(mu)
    out = {"pairs": sets, "scores": {}}
    for n, ps in sets.items():
        sc = {}
        A = E[[ix[prompt(o, a)] for o, a, b in ps]]
        B = E[[ix[prompt(o, b)] for o, a, b in ps]]
        for name, sp in spaces.items():
            a, b = (A, B) if sp is None else (sp.text(A), sp.text(B))
            sc[name] = (a * b).sum(1).tolist()
        if all(o in axes.axes for o, _, _ in ps):  # contradiction axes exist only for MOSCATO objects
            sc["axes-proj"] = [float(axes.text(A[k:k + 1], o)[0] @ axes.text(B[k:k + 1], o)[0])
                               for k, (o, _, _) in enumerate(ps)]
        for c in calls[n]:
            name, s = c.get()
            sc[name] = s
        out["scores"][n] = sc
    json.dump(out, open(eval_sep.SCORES, "w"))
    print("wrote", eval_sep.SCORES)


@app.local_entrypoint()
def food_topup():
    """Score the GPT-authored food additions; preserve original mixed-set caches."""
    import numpy as np
    sys.path.insert(0,HERE)
    import eval_food
    from corpus import prompt
    from space import text_mean
    root=eval_food.OUT
    new=json.loads((root/'food_topup_draft.json').read_text())
    triples=sorted({(k.split('|')[0],k.split('|',1)[1],p)
                    for k,v in new.items() for p in v['same']+[c for _,c in v['contra']]})
    ts=sorted({prompt(o,x) for o,a,b in triples for x in (a,b)})
    print('scoring',len(new),'new anchors,',len(triples),'phrase pairs',flush=True)
    calls=[sep_bi.spawn(n,triples) for n in SEP_BI]
    sig=siglip_texts.spawn(ts)
    mu=text_mean()
    E=np.frombuffer(sig.get(),np.float16).reshape(len(ts),-1).astype(np.float32)-mu
    E/=np.linalg.norm(E,axis=1,keepdims=True)
    ix={t:i for i,t in enumerate(ts)}
    sc={'siglip-centred':[float(E[ix[prompt(o,a)]] @ E[ix[prompt(o,b)]]) for o,a,b in triples]}
    for c in calls:
        name,values=c.get();sc[name]=values
    original=json.loads((root/'sep_scores.json').read_text())
    old=original['scores']['epic-big']
    out={'pairs':{'moscato-big':original['pairs']['moscato-big'],
                  'epic-big':original['pairs']['epic-big']+triples},
         'scores':{'moscato-big':original['scores']['moscato-big'],
                   'epic-big':{name:old[name]+values for name,values in sc.items()}}}
    (root/'food_expanded_scores.json').write_text(json.dumps(out))
    print('wrote',root/'food_expanded_scores.json',flush=True)


@app.local_entrypoint()
def food_short_topup():
    """Score synthetic short food labels and merge with existing food scores."""
    import numpy as np
    sys.path.insert(0,HERE)
    import eval_food
    from corpus import prompt
    from space import text_mean
    root=eval_food.OUT
    new=json.loads((root/'moscato_food_additions.json').read_text())
    triples=sorted({(k.split('|')[0],k.split('|',1)[1],p)
                    for k,v in new.items() for p in v['same']+[c for _,c in v['contra']]})
    ts=sorted({prompt(o,x) for o,a,b in triples for x in (a,b)})
    print('scoring',len(new),'short food labels,',len(triples),'phrase pairs',flush=True)
    calls=[sep_bi.spawn(n,triples) for n in SEP_BI]
    sig=siglip_texts.spawn(ts)
    mu=text_mean()
    E=np.frombuffer(sig.get(),np.float16).reshape(len(ts),-1).astype(np.float32)-mu
    E/=np.linalg.norm(E,axis=1,keepdims=True)
    ix={t:i for i,t in enumerate(ts)}
    sc={'siglip-centred':[float(E[ix[prompt(o,a)]] @ E[ix[prompt(o,b)]]) for o,a,b in triples]}
    for c in calls:
        name,values=c.get();sc[name]=values
    dest=root/'food_expanded_scores.json'
    out=json.loads(dest.read_text())
    original=json.loads((root/'sep_scores.json').read_text())
    out['pairs']['moscato-big']=original['pairs']['moscato-big']+triples
    out['scores']['moscato-big']={name:original['scores']['moscato-big'][name]+values for name,values in sc.items()}
    dest.write_text(json.dumps(out))
    print('wrote',dest,flush=True)


@app.local_entrypoint()
def foods_expanded():
    """Score both expanded food sets, retaining original cached pair scores."""
    import numpy as np
    sys.path.insert(0,HERE)
    import eval_food
    from corpus import prompt
    from space import text_mean
    root=eval_food.OUT
    sets={}
    for name,file in [('moscato-big','moscato_food_additions.json'),('epic-big','food_topup_draft.json')]:
        data=json.loads((root/file).read_text())
        sets[name]=sorted({(k.split('|')[0],k.split('|',1)[1],p)
                          for k,v in data.items() for p in v['same']+[c for _,c in v['contra']]})
    triples=sorted(set().union(*(set(v) for v in sets.values())))
    ts=sorted({prompt(o,x) for o,a,b in triples for x in (a,b)})
    print('scoring expanded food sets:',{k:len(v) for k,v in sets.items()},flush=True)
    calls=[sep_bi.spawn(n,triples) for n in SEP_BI]
    sig=siglip_texts.spawn(ts)
    mu=text_mean()
    E=np.frombuffer(sig.get(),np.float16).reshape(len(ts),-1).astype(np.float32)-mu
    E/=np.linalg.norm(E,axis=1,keepdims=True)
    ix={t:i for i,t in enumerate(ts)}
    sc={'siglip-centred':[float(E[ix[prompt(o,a)]] @ E[ix[prompt(o,b)]]) for o,a,b in triples]}
    for c in calls:
        name,values=c.get();sc[name]=values
    original=json.loads((root/'sep_scores.json').read_text())
    ti={t:i for i,t in enumerate(triples)}
    out={'pairs':{},'scores':{}}
    for name,pairs in sets.items():
        # Remove any duplicate cached pair before appending (the index then stays unambiguous).
        fresh=[t for t in pairs if t not in {tuple(p) for p in original['pairs'][name]}]
        out['pairs'][name]=original['pairs'][name]+fresh
        out['scores'][name]={m:original['scores'][name][m]+[values[ti[t]] for t in fresh] for m,values in sc.items()}
    dest=root/'food_expanded_scores.json'
    dest.write_text(json.dumps(out))
    print('wrote',dest,flush=True)
