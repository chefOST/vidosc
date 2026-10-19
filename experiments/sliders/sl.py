"""Slider exploration (scratch). Zero-shot: prompts written from general cooking knowledge.
MOSCATO GT used only in EVAL map below."""
import os, sys, json, numpy as np
TMP=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(os.path.dirname(TMP))
sys.path.insert(0, ROOT)
from pipeline import data
ds=data.Moscato(ROOT+'/../chefOST-data/MOSCATO', cache_dir=ROOT+'/data/cache')

# slider: low anchors, high anchors, action prompts that move it, monotone-up?
SL={
 'fullness': (["an empty {o}", "the {o} with nothing in it"],
              ["a {o} full of food", "the {o} filled to the top"],
              ["someone pouring something into the {o}", "someone putting food into the {o}"], False),
 'dirt':     (["a clean {o}", "a spotless {o}"],
              ["a dirty {o} covered in food", "a messy {o} smeared with food"],
              ["someone using the {o}", "someone stirring with the {o}"], True),
 'mixed':    (["{o} with separate unmixed ingredients", "whole unmixed {o}"],
              ["{o} whisked into a smooth uniform mixture", "well stirred {o}"],
              ["someone whisking the {o}", "someone stirring the {o}"], True),
 'done':     (["raw uncooked {o}", "fresh raw {o}"],
              ["fully cooked {o}", "golden brown cooked {o}"],
              ["the {o} cooking on heat", "the {o} frying in a pan"], True),
 'cut':      (["a whole uncut {o}", "an intact {o}"],
              ["{o} chopped into small pieces", "finely cut {o}"],
              ["someone cutting the {o} with a knife", "someone chopping the {o}"], True),
 'brown':    (["pale {o}", "{o} with no browning"],
              ["golden brown {o}", "browned crispy {o}"],
              ["the {o} browning in a hot pan", "the {o} baking in the oven"], True),
 'open':     (["a sealed intact {o}", "an unopened {o}"],
              ["an opened {o}", "a cracked open {o}"],
              ["someone opening the {o}", "someone cracking the {o}"], True),
 'heat':     (["a cold {o} that is turned off", "an unheated {o}"],
              ["a hot {o} that is turned on", "a heated {o}"],
              ["someone turning on the {o}", "the {o} heating up"], False),
}
# EVAL ONLY: MOSCATO word -> slider level (eval-side choice, to confirm with user)
EVAL={
 'fullness': {'empty':0,'partially-empty':.5,'full':1,'stuffed':1},
 'dirt': {'clean':0, **{w:1 for w in ['dirty','coated','covered','egg-coated','egg-covered','batter-covered',
          'vegetable-covered','saucy','creamy','sticky','cheese-covered']}},
 'mixed': {'mixed':1,'beaten':1,'mayonnaise-mixed':1,'smooth':1,'cracked':0},
 'done': {'cooked':1,'solid':1,'ready-to-cook':0},
 'cut': {'chopped':1,'grated':1,'peeled':0},
 'heat': {'on':1,'warm':1},
 'brown': {},
 'open': {},
}
def gt_level(states, s):
    m=EVAL[s]; v=[m[w] for w in states if w in m]
    if not v: return None
    if s=='mixed' and 1 in v: return 1   # cracked+beaten -> beaten
    if s=='cut' and 1 in v: return 1
    return max(v) if s!='fullness' else (1 if 1 in v else min(v))

ENC='internvideo-mm-l14'
def feats(vid, enc=None):
    x=np.load(f'{ROOT}/data/cache/emb/{enc or ENC}/{vid}_fps1.npy').astype(np.float32)
    return x/np.linalg.norm(x,axis=1,keepdims=True)

def texts(extra=(), enc=None):
    """Text embeddings for all slider prompts x objects (+ extra templates), cached per encoder."""
    enc=enc or ENC
    objs=sorted({o for v in ds.videos for o in ds.objects(v)})
    tpl=[t for s,(lo,hi,ac,_) in SL.items() for t in lo+hi+ac]+list(extra)
    allt=sorted({t.format(o=o) for o in objs for t in tpl})
    npz=f'{TMP}/text.npz' if enc=='internvideo-mm-l14' else f'{TMP}/text_{enc}.npz'
    emb={}
    if os.path.exists(npz):
        d=np.load(npz,allow_pickle=True); emb=dict(zip(d['t'],d['e']))
    need=[t for t in allt if t not in emb]
    if need:
        if enc=='internvideo-mm-l14':
            from pipeline.internvideo import InternVideoEncoder; m=InternVideoEncoder()
        else:
            from pipeline.encoders import CLIPEncoder
            name=enc[len('clip-'):]; arch,pre=name.rsplit('-',1) if '/' not in name else name.split('/')
            m=CLIPEncoder(arch+'/'+pre)
        e=np.concatenate([m.encode_texts(need[i:i+64]) for i in range(0,len(need),64)])
        emb.update(zip(need,e)); np.savez(npz,t=np.array(list(emb)),e=np.array(list(emb.values())))
    return emb

def mean_t(emb, ts, o):
    v=np.mean([emb[t.format(o=o)] for t in ts],0); return v/np.linalg.norm(v)

ASSIGN={k:v for k,v in json.load(open(os.path.join(TMP,'object_sliders.json'))).items() if not k.startswith('_')}
