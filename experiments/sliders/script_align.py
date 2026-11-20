'Decode LLM slider scripts against visual evidence using dynamic programming.'
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, evidence as B
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
