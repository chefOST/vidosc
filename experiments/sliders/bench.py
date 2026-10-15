"""Slider benchmark on the 30 GT videos, assigned (object, slider) pairs only.
M1 level: concordance of per-frame readout with GT level, pooled over all tracks (0.5 = chance).
M2 timing (tracks whose GT level changes): median |predicted step - first GT-high frame| in seconds,
   and balanced accuracy of the step. 'mid' = step at the middle (no video).
Usage: python bench.py [encoder-name]"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, sl
from collections import defaultdict

EXTRA=["a {o}", "someone holding the {o}", "a photo of the {o}"]
def smooth(x,w=9):
    k=np.ones(w)/w; return np.convolve(np.pad(x,w//2,mode='edge'),k,'valid')
z=lambda x:(x-x.mean())/(x.std()+1e-8)
def conc(p,y):
    p=np.asarray(p,float);y=np.asarray(y,float); n=c=0.
    for a in np.unique(y):
        for b in np.unique(y):
            if a<b:
                pa=np.sort(p[y==a]); pb=p[y==b]
                lt=np.searchsorted(pa,pb,'left'); le=np.searchsorted(pa,pb,'right')
                c+=lt.sum()+.5*(le-lt).sum(); n+=len(pa)*len(pb)
    return c/n if n else np.nan
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
            G=sl.ds.gt_states(vid,o,idx)
            for s in sl.ASSIGN.get(o,[]):
                y=np.array([-1 if (v:=sl.gt_level(g,s)) is None else v for g in G],float)
                if not (y>=0).any(): continue
                lo,hi,ac,mono=sl.SL[s]
                ra=RA.get(vid.split('_')[1],{}).get(o,{}).get(s)
                if anchors=='recipe' and ra: lo,hi,ac=[ra[0]],[ra[1]],[ra[2]]
                if anchors=='both' and ra: lo,hi,ac=lo+[ra[0]],hi+[ra[1]],ac+[ra[2]]
                t=dict(vid=vid,o=o,s=s,y=y,T=T,F=F,mono=mono,
                       d=F@sl.mean_t(emb,hi,o)-F@sl.mean_t(emb,lo,o),
                       a=F@sl.mean_t(emb,ac,o),
                       pres=F@sl.mean_t(emb,EXTRA[:2],o))
                tracks.append(t)
    return tracks,emb

def report(tracks, level_fns, step_fns, sliders=('fullness','dirt','mixed','done','cut','heat')):
    print('M1 level concordance per object (pooled over its videos), frame-weighted mean over objects')
    print(f'{"":22s}'+''.join(f'{s:>9s}' for s in sliders))
    for k,f in level_fns.items():
        row=''
        for s in sliders:
            by=defaultdict(lambda:([],[]))
            for t in tracks:
                if t['s']!=s: continue
                m=t['y']>=0; by[t['o']][0].extend(f(t)[m]); by[t['o']][1].extend(t['y'][m])
            cw=[(conc(*v),len(v[1])) for v in by.values()]; cw=[(c,n) for c,n in cw if not np.isnan(c)]
            row+=f'{sum(c*n for c,n in cw)/sum(n for c,n in cw):9.3f}' if cw else f'{"nan":>9s}'
        print(f'{k:22s}'+row)
    ch=[t for t in tracks if len(set((t['y'][t['y']>=0]>=.5)))>1]
    print(f'\nM2 timing on changing tracks: median abs error s / balanced acc   n=',
          {s:sum(t["s"]==s for t in ch) for s in sliders})
    print(f'{"":22s}'+''.join(f'{s:>14s}' for s in sliders))
    fr=np.linspace(0,1,41)
    row=''
    for s in sliders:
        cs=[t for t in ch if t['s']==s]
        if not cs: row+=f'{"-":>14s}'; continue
        errs=[np.median([abs(int(f_*t['T'])-np.argmax(t['y']>=.5)) for t in cs]) for f_ in fr]
        row+=f'{min(errs):7.0f}@{fr[int(np.argmin(errs))]:.2f}'
    print(f'{"best fixed frac (GT!)":22s}'+row)
    for k,f in step_fns.items():
        row=''
        for s in sliders:
            E,B=[],[]
            for t in ch:
                if t['s']!=s: continue
                m=t['y']>=0; yb=(t['y'][m]>=.5).astype(int); tau=f(t)
                pr=(np.arange(t['T'])>=tau)[m].astype(int)
                E.append(abs(tau-np.argmax(t['y']>=.5))); B.append(.5*((pr[yb==0]==0).mean()+(pr[yb==1]==1).mean()))
            row+=f'{np.median(E):7.0f}/{np.mean(B):.2f}' if E else f'{"-":>14s}'
        print(f'{k:22s}'+row)

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
if __name__=='__main__':
    enc=sys.argv[1] if len(sys.argv)>1 and sys.argv[1]!='-' else None
    tr,_=load(enc, sys.argv[2] if len(sys.argv)>2 else 'generic')
    report(tr,LEVEL,STEP)
