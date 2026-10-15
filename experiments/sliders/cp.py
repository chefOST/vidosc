"""Monotone sliders as a single step: when does the object go low->high? Tracks with both levels."""
import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, sl
from collections import defaultdict
emb=sl.texts()
def smooth(x,w=9):
    k=np.ones(w)/w; return np.convolve(np.pad(x,w//2,mode='edge'),k,'valid')
def contrast(d):
    # mean-shift statistic for a step up at tau (t-stat like), tau in [1,T-1]
    T=len(d); c=np.cumsum(d); out=np.full(T,-np.inf)
    for t in range(2,T-1):
        a=c[t-1]/t; b=(c[-1]-c[t-1])/(T-t); out[t]=(b-a)*np.sqrt(t*(T-t)/T)
    return out
def step(tau,T): return (np.arange(T)>=tau).astype(int)
R=defaultdict(lambda: defaultdict(list))
for vid in sl.ds.videos:
    F=sl.feats(vid); T=len(F); idx=list(range(0,30*T,30))
    for o in sl.ds.objects(vid):
        G=sl.ds.gt_states(vid,o,idx)
        for s,(lo,hi,ac,mono) in sl.SL.items():
            if not mono: continue
            y=np.array([-1 if (v:=sl.gt_level(g,s)) is None else v for g in G]); m=y>=0
            if len(set(y[m]))<2: continue
            d=F@sl.mean_t(emb,hi,o)-F@sl.mean_t(emb,lo,o)
            a=smooth(F@sl.mean_t(emb,ac,o))
            # also: frame-to-frame visual change as a generic "something is happening" cue
            mv=np.r_[0,1-(F[1:]*F[:-1]).sum(1)]; mv=smooth(mv)
            cs=contrast(d)
            acc=lambda tau:(step(tau,T)[m]==y[m]).mean()
            oracle=max(range(T+1),key=acc)
            c2=cs.copy(); c2[~np.isfinite(c2)]=-1e9
            # action gating: only allow the step where the action cue is in its top 30% for this video
            gate=a>=np.quantile(a,.7); cg=np.where(gate,c2,-1e9)
            mg=mv>=np.quantile(mv,.7); cm=np.where(mg,c2,-1e9)
            P={'oracle step':oracle,'mid (no video)':T//2,
               'first-labeled-frac (no video)':None,
               'video contrast':int(np.argmax(c2)),
               'contrast+action gate':int(np.argmax(cg)),
               'contrast+motion gate':int(np.argmax(cm)),
               'action peak':int(np.argmax(a)),
               'all high (no video)':0}
            for k,tau in P.items():
                if tau is None: continue
                R[s][k].append(acc(tau)); R['ALL'][k].append(acc(tau))
            R[s]['_terr'].append(abs(P['video contrast']-oracle)/T); R['ALL']['_terr'].append(abs(P['video contrast']-oracle)/T)
            R['ALL']['_terr_mid'].append(abs(T//2-oracle)/T)
for s,d in R.items():
    n=len(d['mid (no video)'])
    print(f'\n{s} (tracks={n})')
    for k,v in d.items(): print(f'  {k:24s} {np.mean(v):.3f}')
