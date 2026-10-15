"""Fuse two encoders: readout d = z(d_A)+z(d_B) per track; action/presence from A (video model).
Usage: python fuse.py encA encB anchors"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bench as B
a,b,anc=sys.argv[1],sys.argv[2],sys.argv[3]
A,_=B.load(None if a=='-' else a,anc); Bt,_=B.load(b,anc)
for x,y in zip(A,Bt):
    assert (x['vid'],x['o'],x['s'])==(y['vid'],y['o'],y['s'])
    n=min(x['T'],y['T'])
    for k in ['d','pres','a','y']: x[k]=x[k][:n]
    for k in ['d','pres']: y[k]=y[k][:n]
    x['T']=n
    x['d']=B.z(x['d'])+B.z(y['d']); x['pres']=B.z(x['pres'])+B.z(y['pres'])
B.report(A,B.LEVEL,B.STEP)
