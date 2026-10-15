import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, sl
from sklearn.isotonic import IsotonicRegression
from collections import defaultdict
emb=sl.texts()
def sig(x): return 1/(1+np.exp(-x))
def conc(p,y):
    p=np.asarray(p);y=np.asarray(y); n=c=0.
    for a in np.unique(y):
        for b in np.unique(y):
            if a<b:
                pa=p[y==a][:,None]; pb=p[y==b][None,:]
                c+=(pb>pa).sum()+.5*(pb==pa).sum(); n+=pa.size*pb.size
    return c/n if n else None
def smooth(x,w=9):
    k=np.ones(w)/w; return np.convolve(np.pad(x,w//2,mode='edge'),k,'valid')
def iso(x): return IsotonicRegression().fit_transform(np.arange(len(x)),x)
def integ(a):
    z=(a-a.mean())/(a.std()+1e-6); c=np.cumsum(np.maximum(z,0)); return c/(c[-1]+1e-6)
DEC={
 'raw': lambda r,a,mono: r,
 'smooth': lambda r,a,mono: smooth(r),
 'mono': lambda r,a,mono: iso(smooth(r)) if mono else smooth(r),
 'ramp(no video)': lambda r,a,mono: np.linspace(0,1,len(r)),
 'action-integral': lambda r,a,mono: integ(smooth(a)),
}
res=defaultdict(lambda: defaultdict(list)); pooled=defaultdict(lambda: defaultdict(lambda: ([],[])))
for vid in sl.ds.videos:
    F=sl.feats(vid); idx=list(range(0,30*len(F),30))
    for o in sl.ds.objects(vid):
        G=sl.ds.gt_states(vid,o,idx)
        for s,(lo,hi,ac,mono) in sl.SL.items():
            y=[sl.gt_level(g,s) for g in G]; m=np.array([v is not None for v in y])
            if not m.any(): continue
            yy=np.array([v for v in y if v is not None])
            r=sig(100*(F@sl.mean_t(emb,hi,o)-F@sl.mean_t(emb,lo,o)))
            a=F@sl.mean_t(emb,ac,o)
            for d,f in DEC.items():
                p=f(r,a,mono)[m]
                pooled[s][d][0].extend(p); pooled[s][d][1].extend(yy)
                c=conc(p,yy)
                if c is not None: res[s][d].append(c)
print('concordance (0.5=chance). pooled = across all videos; within = mean over tracks that change')
for s in sl.SL:
    print(f'\n{s}: tracks-with-change={len(res[s]["raw"])} frames={len(pooled[s]["raw"][0])} pos-rate={np.mean(pooled[s]["raw"][1]):.2f}')
    for d in DEC:
        pc=conc(*pooled[s][d]); w=np.mean(res[s][d]) if res[s][d] else float('nan')
        print(f'  {d:16s} pooled {pc if pc is not None else float("nan"):.3f}  within {w:.3f}')
