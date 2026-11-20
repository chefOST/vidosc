'Slider evidence extraction and temporal inference helpers. No annotation scoring.'
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, sl
from collections import defaultdict

EXTRA=["a {o}", "someone holding the {o}", "a photo of the {o}"]
def smooth(x,w=9):
    k=np.ones(w)/w; return np.convolve(np.pad(x,w//2,mode='edge'),k,'valid')
z=lambda x:(x-x.mean())/(x.std()+1e-8)
def step_contrast(e):
    T=len(e); e=z(e)/np.sqrt(T); c=np.r_[0,np.cumsum(e)]
    return int(np.argmax((c[-1]-c[:T])-c[:T]))

import json
RA={k:v for k,v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'recipe_anchors.json'))).items() if k[0]!='_'}
RTEXT=sorted({x for r in RA.values() for o in r.values() for a in o.values() for x in a})
def load(enc=None, anchors='generic'):
    emb=sl.texts(EXTRA+RTEXT,enc); tracks=[]
    for vid in sl.ds.videos:
        F=sl.feats(vid,enc); T=len(F); idx=list(range(0,30*T,30))
        for o in sl.ds.objects(vid):
            for s in sl.ASSIGN.get(o,[]):
                lo,hi,ac,mono=sl.SL[s]
                ra=RA.get(vid.split('_')[1],{}).get(o,{}).get(s)
                if anchors=='recipe' and ra: lo,hi,ac=[ra[0]],[ra[1]],[ra[2]]
                if anchors=='both' and ra: lo,hi,ac=lo+[ra[0]],hi+[ra[1]],ac+[ra[2]]
                t=dict(vid=vid,o=o,s=s,T=T,F=F,mono=mono,
                       d=F@sl.mean_t(emb,hi,o)-F@sl.mean_t(emb,lo,o),
                       a=F@sl.mean_t(emb,ac,o),
                       pres=F@sl.mean_t(emb,EXTRA[:2],o))
                tracks.append(t)
    return tracks,emb


def vis_w(t,q=.7):
    p=smooth(t['pres']); return (p>=np.quantile(p,q)).astype(float)
def hold(t,lam=.97,q=.7):
    # causal: slider updates only on frames where the object is visible, else holds
    w=vis_w(t,q); d=t['d']; num=den=0.; out=np.empty(len(d))
    for i in range(len(d)):
        num=lam*num+w[i]*d[i]; den=lam*den+w[i]; out[i]=num/den if den>1e-6 else d[i]
    return out
def hold2(t,sig=10,q=.7):
    # two-sided visibility-weighted smoothing
    w=vis_w(t,q); g=np.exp(-.5*(np.arange(-3*sig,3*sig+1)/sig)**2)
    num=np.convolve(w*t['d'],g,'same'); den=np.convolve(w,g,'same')
    return np.where(den>1e-3,num/np.maximum(den,1e-3),smooth(t['d']))
from sklearn.isotonic import IsotonicRegression
def iso(x): return IsotonicRegression().fit_transform(np.arange(len(x)),x)
LEVEL={'ramp t/T (no video)': lambda t: np.arange(t['T'])/t['T'],
       'readout': lambda t: t['d'], 'readout smooth': lambda t: smooth(t['d']),
       'visible-hold causal': hold, 'visible-hold 2-sided': hold2,
       'ramp + vis-hold (z sum)': lambda t: z(np.arange(t['T'])/t['T'])*t['mono']+z(hold2(t)),
       'vis-hold + one-way': lambda t: iso(hold2(t)) if t['mono'] else hold2(t)}
STEP={'mid (no video)': lambda t: t['T']//2,
      'readout contrast': lambda t: step_contrast(smooth(t['d'])),
      'action peak': lambda t: int(np.argmax(smooth(t['a']))),
      'contrast on vis-hold': lambda t: step_contrast(hold2(t)),
      'presence onset': lambda t: int(np.argmax(z(smooth(t['pres']))>1))}
