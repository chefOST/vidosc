"""LLM script (ordered slider levels) + video timing. DP splits each track into the script's segments,
maximizing sum_t (level_k - .5) * e_t, e = z-scored slider evidence. Optional boundary bonus from an action cue.
Eval: per-frame range hit, (pred>=.5)==(GT>=.5); balanced acc pooled per slider; also frame acc.
Usage: python script_align.py [enc|-] [anchors] [fuse-enc]"""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bench as B
from collections import defaultdict
SC={k:v for k,v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'scripts.json'))).items() if k[0]!='_'}

def align(levels, e, bonus=None, minlen=3):
    """Best split of e (length T) into len(levels) contiguous segments. Returns per-frame level."""
    T,K=len(e),len(levels)
    if K==1: return np.full(T,levels[0],float)
    c=np.r_[0,np.cumsum(e)]
    NEG=-1e18; D=np.full((K,T+1),NEG); arg=np.zeros((K,T+1),int)
    for t in range(minlen,T+1): D[0,t]=(levels[0]-.5)*c[t]
    for k in range(1,K):
        w=levels[k]-.5
        for t in range((k+1)*minlen,T+1):
            s=np.arange(k*minlen,t-minlen+1)            # start of segment k
            v=D[k-1,s]+w*(c[t]-c[s])+(bonus[s] if bonus is not None else 0)
            j=int(np.argmax(v)); D[k,t]=v[j]; arg[k,t]=s[j]
    out=np.empty(T); t=T
    for k in range(K-1,-1,-1):
        s=arg[k,t] if k>0 else 0; out[s:t]=levels[k]; t=s
    return out

def run(tracks, methods):
    R=defaultdict(lambda:defaultdict(lambda:[[],[]])); cover=defaultdict(lambda:[0,0])
    for t in tracks:
        sc=SC.get(t['vid'].split('_')[1],{}).get(t['o'],{}).get(t['s'])
        cover[t['s']][1]+=1
        if sc is None: continue
        cover[t['s']][0]+=1
        m=t['y']>=0; yb=t['y'][m]>=.5
        for k,f in methods.items():
            p=f(t,sc)[m]>=.5; R[t['s']][k][0].extend(p); R[t['s']][k][1].extend(yb)
    ss=[s for s in ['fullness','dirt','mixed','done','cut','heat'] if s in R]
    print('coverage (tracks with a script / evaluable):',{s:f'{cover[s][0]}/{cover[s][1]}' for s in ss})
    print('range hit: balanced acc / frame acc (pooled frames)')
    print(f'{"":30s}'+''.join(f'{s:>13s}' for s in ss))
    for k in methods:
        row=''
        for s in ss:
            p,y=map(np.array,R[s][k]); fa=(p==y).mean()
            ba=.5*(p[y].mean()+(~p[~y]).mean()) if y.any() and (~y).any() else float('nan')
            row+=f'{ba:7.3f}/{fa:.3f}'
        print(f'{k:30s}'+row)

ev=lambda t:B.z(B.hold2(t))
M={'script, equal split (no video)': lambda t,sc: align(sc,np.zeros(t['T'])) if len(sc)==1 else
       np.array(sc,float)[np.minimum((np.arange(t['T'])*len(sc))//t['T'],len(sc)-1)],
   'no script: vis-hold d>0': lambda t,sc: (B.hold(t)>0).astype(float),
   'script + readout': lambda t,sc: align(sc,B.z(B.smooth(t['d']))),
   'script + vis-hold readout': lambda t,sc: align(sc,ev(t)),
   'script + vis-hold + action': lambda t,sc: align(sc,ev(t),bonus=2*B.z(B.smooth(t['a']))),
   'script + vis-hold + handled': lambda t,sc: align(sc,ev(t),bonus=2*B.z(B.smooth(t['pres']))),
   'script + handled only': lambda t,sc: align(sc,np.zeros(t['T']),bonus=B.z(B.smooth(t['pres']))),
}
if __name__=='__main__':
    enc=None if len(sys.argv)<2 or sys.argv[1]=='-' else sys.argv[1]
    anc=sys.argv[2] if len(sys.argv)>2 else 'both'
    tr,_=B.load(enc,anc)
    if len(sys.argv)>3:
        tb,_=B.load(sys.argv[3],anc)
        for x,y in zip(tr,tb):
            n=min(x['T'],y['T'])
            for k in ['d','pres','a','y']: x[k]=x[k][:n]
            x['T']=n; x['d']=B.z(x['d'])+B.z(y['d'][:n]); x['pres']=B.z(x['pres'])+B.z(y['pres'][:n])
    run(tr,M)
