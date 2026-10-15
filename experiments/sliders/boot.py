"""Bootstrap over videos: balanced range-hit acc of script+video vs script equal split (no video)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bench as B, script_align as S
from collections import defaultdict
tr,_=B.load(None,'both')
P=defaultdict(dict)
for t in tr:
    sc=S.SC.get(t['vid'].split('_')[1],{}).get(t['o'],{}).get(t['s'])
    if sc is None: continue
    m=t['y']>=0; y=t['y'][m]>=.5
    P[t['s']].setdefault(t['vid'],[]).append((y, S.M['script + vis-hold readout'](t,sc)[m]>=.5,
                                               S.M['script, equal split (no video)'](t,sc)[m]>=.5))
def bal(rows,i):
    y=np.concatenate([r[0] for r in rows]); p=np.concatenate([r[i] for r in rows])
    return .5*(p[y].mean()+(~p[~y]).mean()) if y.any() and (~y).any() else (p==y).mean()  # frame acc if one class
rng=np.random.default_rng(0)
for s,byv in P.items():
    vids=list(byv); full=[r for v in vids for r in byv[v]]
    d=[]
    for _ in range(1000):
        rows=[r for v in rng.choice(vids,len(vids)) for r in byv[v]]; d.append(bal(rows,1)-bal(rows,2))
    d=np.array(d)
    print(f'{s:9s} videos={len(vids):2d}  video {bal(full,1):.3f} vs no-video {bal(full,2):.3f}  diff 95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]')
