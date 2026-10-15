"""One-way 2-state model per track: state 0 -> 1 once, at tau.
score(tau) = beta * sum_t emis_t(state) + gamma * z(action)[tau]   (emission from slider readout z-scored in-track)
Hyperparameters fixed a priori (beta=gamma=1); sweep printed only as sensitivity (tuned on GT -> not a result)."""
import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, sl
from collections import defaultdict
emb=sl.texts()
def smooth(x,w=9):
    k=np.ones(w)/w; return np.convolve(np.pad(x,w//2,mode='edge'),k,'valid')
z=lambda x:(x-x.mean())/(x.std()+1e-8)
def best_tau(d,a,beta,gamma,shift=0):
    T=len(d); e=z(smooth(d))/np.sqrt(T)       # per-frame evidence for "high"
    # sum_{t>=tau} e_t - sum_{t<tau} e_t
    c=np.r_[0,np.cumsum(e)]; tot=c[-1]
    s=beta*((tot-c[:T])-c[:T]) + gamma*np.roll(z(a),shift)[:T]
    return int(np.argmax(s))
TRACKS=[]
for vid in sl.ds.videos:
    F=sl.feats(vid); T=len(F); idx=list(range(0,30*T,30))
    for o in sl.ds.objects(vid):
        G=sl.ds.gt_states(vid,o,idx)
        for s,(lo,hi,ac,mono) in sl.SL.items():
            y=np.array([-1 if (v:=sl.gt_level(g,s)) is None else int(v>=.5) for g in G]); m=y>=0
            if len(set(y[m]))<2: continue
            d=F@sl.mean_t(emb,hi,o)-F@sl.mean_t(emb,lo,o); a=smooth(F@sl.mean_t(emb,ac,o))
            TRACKS.append((vid,o,s,d,a,y,m))
ALLT=TRACKS
TRACKS=[t for t in ALLT if t[2] in sl.ASSIGN.get(t[1],[])]
print('coverage: changing GT tracks kept by the object->slider assignment:',len(TRACKS),'/',len(ALLT))
for t in ALLT:
    if t not in TRACKS: print('  dropped:',t[1],t[2])
def evaluate(beta,gamma,shift=0,which=None):
    out=defaultdict(list)
    for vid,o,s,d,a,y,m in TRACKS:
        if which and s not in which: continue
        T=len(d); tau=best_tau(d,a,beta,gamma,shift)
        out[s].append(((np.arange(T)>=tau)[m]==y[m]).mean())
    return out
def base(kind):
    out=defaultdict(list)
    for vid,o,s,d,a,y,m in TRACKS:
        T=len(d); tau={'mid':T//2,'allhigh':0}[kind]
        out[s].append(((np.arange(T)>=tau)[m]==y[m]).mean())
    return out
rows={'mid (no video)':base('mid'),'all-high (no video)':base('allhigh'),
      'slider only (b1,g0)':evaluate(1,0),'action only (b0,g1)':evaluate(0,1),
      'slider+action (b1,g1)':evaluate(1,1)}
ss=sorted({t[2] for t in TRACKS})
print('frame acc on tracks that change; n tracks:',{s:sum(t[2]==s for t in TRACKS) for s in ss})
print(f'{"":24s}'+''.join(f'{s:>10s}' for s in ss))
for k,r in rows.items(): print(f'{k:24s}'+''.join(f'{np.mean(r[s]):10.3f}' for s in ss))
print('\nsensitivity (GT-tuned, not a result): mixed/dirt/fullness')
for b in [0,.5,1,2,4]:
    for g in [0,.5,1,2]:
        r=evaluate(b,g); print(f'  b={b} g={g}: '+' '.join(f'{s}={np.mean(r[s]):.3f}' for s in ['mixed','dirt','fullness']))

print('\n=== balanced acc (mean of low-frame acc and high-frame acc; all-high = 0.5) ===')
def bal(pred,y): return .5*((pred[y==0]==0).mean()+(pred[y==1]==1).mean())
def run(rule):
    out=defaultdict(list)
    for vid,o,s,d,a,y,m in TRACKS:
        T=len(d); tau=rule(d,a,T); out[s].append(bal((np.arange(T)>=tau)[m].astype(int),y[m]))
    return out
def onset(a,T,th=1.0):
    w=np.where(z(a)>th)[0]; return int(w[0]) if len(w) else T//2
R={'mid (no video)':run(lambda d,a,T:T//2),
   'slider only':run(lambda d,a,T:best_tau(d,a,1,0)),
   'action peak':run(lambda d,a,T:best_tau(d,a,0,1)),
   'action onset z>1':run(lambda d,a,T:onset(a,T)),
   'slider+action':run(lambda d,a,T:best_tau(d,a,1,1))}
print(f'{"":24s}'+''.join(f'{s:>10s}' for s in ss))
for k,r in R.items(): print(f'{k:24s}'+''.join(f'{np.mean(r[s]):10.3f}' for s in ss))
from scipy.stats import wilcoxon
for s in ['mixed','fullness','dirt']:
    for k in ['action peak','action onset z>1','slider only']:
        a_=np.array(R[k][s]); b_=np.array(R['mid (no video)'][s])
        print(f'{s} {k} vs mid: wins {np.sum(a_>b_)}/{len(a_)}  p={wilcoxon(a_,b_).pvalue:.3f}')
print('\nmixed tracks:')
for (vid,o,s,d,a,y,m),acc in zip([t for t in TRACKS if t[2]=='mixed'],R['action peak']['mixed']):
    T=len(d); print(f'  {vid:28s} {o:14s} T={T} gt_first_high={np.argmax((y==1))} action_peak={best_tau(d,a,0,1)} bal={acc:.2f}')

print('\n=== timing error |pred step - first GT high| in seconds (median) ===')
for s in ['mixed','fullness','dirt']:
    for rec in ['Eggs','Brownie','Pizza','Salad','Sandwich',None]:
        tr=[t for t in TRACKS if t[2]==s and (rec is None or rec in t[0])]
        if not tr: continue
        e_a=[abs(best_tau(d,a,0,1)-np.argmax(y==1)) for _,_,_,d,a,y,m in tr]
        e_m=[abs(len(d)//2-np.argmax(y==1)) for _,_,_,d,a,y,m in tr]
        e_s=[abs(best_tau(d,a,1,0)-np.argmax(y==1)) for _,_,_,d,a,y,m in tr]
        print(f'  {s:9s} {rec or "ALL":9s} n={len(tr):2d}  action {np.median(e_a):5.0f}s  slider {np.median(e_s):5.0f}s  mid {np.median(e_m):5.0f}s')
