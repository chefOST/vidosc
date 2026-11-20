"""One-way two-state temporal decoder; no ground-truth scoring."""
import numpy as np
z=lambda x:(x-x.mean())/(x.std()+1e-8)

def smooth(x,w=9):
    k=np.ones(w)/w; return np.convolve(np.pad(x,w//2,mode='edge'),k,'valid')

def best_tau(d,a,beta,gamma,shift=0):
    T=len(d); e=z(smooth(d))/np.sqrt(T)       # per-frame evidence for "high"
    # sum_{t>=tau} e_t - sum_{t<tau} e_t
    c=np.r_[0,np.cumsum(e)]; tot=c[-1]
    s=beta*((tot-c[:T])-c[:T]) + gamma*np.roll(z(a),shift)[:T]
    return int(np.argmax(s))

def onset(a,T,th=1.0):
    w=np.where(z(a)>th)[0]; return int(w[0]) if len(w) else T//2
