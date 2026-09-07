import json, sys, numpy as np, torch
sys.argv=["x"]; src=open("scripts/train_chaos.py").read()
exec(src[src.index("def features"):src.index("def rand_rot")])
d=np.load("data/three_body_3d_fresh3k.npz"); POS=d["pos"]; n=len(POS); T=float(d["times"][-1])
last=(~np.isnan(POS[:,:,0,0])).sum(1)-1; rlast=np.linalg.norm(POS[np.arange(n),last],axis=2).max(1); st=d["status"]
Y=np.stack([((d["t_div"]<T)&(st==0)), rlast>20, st==1],1).astype(float); W=np.stack([st==0, np.ones(n), np.ones(n)],1)
M=json.load(open("data/chaos_model.json")); F=features(d["masses"],POS[:,0],d["vel"][:,0])
h=(F-np.array(M["mu"]))/np.array(M["sd"])
for l,(Wt,b) in enumerate(zip(M["W"],M["b"])):
    z=h@np.array(Wt).T+np.array(b); h=z/(1+np.exp(-z)) if l<len(M["W"])-1 else z
P=1/(1+np.exp(-h))
def auc(y,p):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1); npos=y.sum(); nneg=len(y)-npos
    return (r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg)
for k,name in enumerate(["diverge","escape","collision"]):
    sel=W[:,k]>0; print(f"fresh 3000 (never used): {name:9s} n={sel.sum()} AUC {auc(Y[sel,k],P[sel,k]):.3f}   (original test set: {M['results']['mlp_aug'][name]['auc']:.3f})")
