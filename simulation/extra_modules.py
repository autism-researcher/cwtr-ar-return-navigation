"""
Component evaluations appended to the CWTR study (all seeded, numpy-only):
ablation, wandering detection (ROC/AUC/F1), confidence-gated false-trigger,
behaviour-adaptive bandit cueing (illustrative model), profile cost matrix,
and runtime. Prints exactly the numbers quoted in the paper.

Reproduce:  python3 extra_modules.py     (run from this directory)
"""
import numpy as np, time
import cwtr_simulation as S   # the validated reconstruction harness

# ---------------- Ablation (primary operating point) ----------------
def cwtr_rmse(rate, use_acc=True, use_kin=True, seed=S.SEED):
    r = np.random.default_rng(seed); out=[]
    for _ in range(S.M):
        truth=S.gen_truth(r); z,a,sigma=S.gen_obs(truth,rate,r); n=len(z)
        c=S.conf(z,a,use_acc=use_acc,use_kin=use_kin)   # canonical confidence
        Rc=[np.eye(2)*((S.SIGMA0**2/c[t])*(S.GATE if c[t]<S.TAU else 1.0)) for t in range(n)]
        out.append(S.metrics(S.kalman_rts(z,Rc),truth)[0])
    return float(np.mean(out))
RATE=S.PRIMARY_RATE

def _print_ablation():
    print("=== ABLATION (primary rate=%.2f) ===" % RATE)
    print(f"full CWTR RMSE   = {cwtr_rmse(RATE,True,True):.2f} m")
    print(f"remove kinematic = {cwtr_rmse(RATE,True,False):.2f} m")
    print(f"remove accuracy  = {cwtr_rmse(RATE,False,True):.2f} m")

# ---------------- Trajectory generators + wandering score ----------------
def gen_purposeful(r, hsd=0.18):  # hsd=0.18: realistic goal-directed (classification);
    h=r.uniform(0,2*np.pi); p=np.zeros(2); P=[p.copy()]   # hsd=0.08: straight walks (false-alarm test)
    for _ in range(120):
        h+=r.normal(0,hsd); p=p+1.3*np.array([np.cos(h),np.sin(h)]); P.append(p.copy())
    return np.array(P)
def gen_wandering(r):
    h=r.uniform(0,2*np.pi); p=np.zeros(2); P=[p.copy()]
    for _ in range(120):
        h+=r.normal(0,0.55); p=p+r.uniform(0.4,1.3)*np.array([np.cos(h),np.sin(h)]); P.append(p.copy())
    return np.array(P)
def wscore(pos):
    """Irregularity score S_W exactly as Eq. (6) of the paper:
    S_W = a1*(1-SR) + a2*sigma_dtheta_norm + a3*r_norm,
    (a1,a2,a3)=(0.5,0.3,0.2), each term normalized to [0,1],
    revisit counted on the 5 m grid stated in the paper."""
    A1, A2, A3 = 0.5, 0.3, 0.2
    d=np.diff(pos,axis=0); L=np.linalg.norm(d,axis=1).sum()
    net=np.linalg.norm(pos[-1]-pos[0]); SR=net/(L+1e-9)          # straightness in (0,1]
    ang=np.arctan2(d[:,1],d[:,0]); dh=np.abs(np.diff(ang)); dh=np.minimum(dh,2*np.pi-dh)
    sig=min(1.0, (np.std(dh)/np.pi)) if len(dh) else 0.0         # heading-change SD, normalized
    cells={tuple(x) for x in np.round(pos/5.0).astype(int)}      # 5 m grid (Sec. V)
    r_norm=(len(pos)-len(cells))/len(pos)                        # revisit fraction in [0,1)
    return A1*(1-SR) + A2*sig + A3*r_norm

# ---------------- Wandering detection ----------------
def wandering_experiment(seed=11, n=250):
    """Returns (auc, bestf1, prec, rec, acc, sp, sw) computed live."""
    r=np.random.default_rng(seed)
    sp=np.array([wscore(gen_purposeful(r)) for _ in range(n)])
    sw=np.array([wscore(gen_wandering(r)) for _ in range(n)])
    scores=np.concatenate([sp,sw]); labels=np.array([0]*n+[1]*n)
    order=np.argsort(scores); ranks=np.empty(len(scores)); ranks[order]=np.arange(1,len(scores)+1)
    n1=labels.sum(); n0=len(labels)-n1
    auc=(ranks[labels==1].sum()-n1*(n1+1)/2)/(n1*n0)
    bestf1=0; bs=None
    for th in np.unique(scores):
        pred=(scores>=th).astype(int)
        tp=((pred==1)&(labels==1)).sum(); fp=((pred==1)&(labels==0)).sum(); fn=((pred==0)&(labels==1)).sum()
        prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9)
        if f1>bestf1: bestf1=f1; bs=(prec,rec,pred)
    prec,rec,pred=bs; acc=(pred==labels).mean()
    return auc,bestf1,prec,rec,acc,sp,sw

# ---------------- Confidence-gated false-trigger ----------------
def confgated_experiment(seed_thr=23, seed_run=24, N=200):
    """Returns (raw_pct, cwtr_pct) false-trigger rates computed live."""
    r=np.random.default_rng(seed_thr)
    clean=[wscore(gen_purposeful(r,0.08)) for _ in range(N)]  # straight goal-directed
    thr=np.percentile(clean,95)
    raw_flag=cwtr_flag=0; r=np.random.default_rng(seed_run)
    for _ in range(N):
        truth=gen_purposeful(r,0.08); Tn=len(truth)
        z=truth+r.normal(0,4.0,(Tn,2))
        for t in range(Tn):
            if r.random()<0.12:
                z[t]=truth[t]+r.uniform(15,55)*np.array([np.cos(r.uniform(0,2*np.pi)),np.sin(r.uniform(0,2*np.pi))])
        a=np.full(Tn,4.0)
        if wscore(z)>=thr: raw_flag+=1
        c=S.conf(z,a)   # canonical confidence (uniform 1 Hz -> c_tmp inactive)
        Rc=[np.eye(2)*((S.SIGMA0**2/c[t])*(S.GATE if c[t]<S.TAU else 1.0)) for t in range(Tn)]
        if wscore(S.kalman_rts(z,Rc))>=thr: cwtr_flag+=1
    return 100*raw_flag/N, 100*cwtr_flag/N



def _print_detector_sections():
    auc,bestf1,prec,rec,acc,sp,sw = wandering_experiment()
    print("\n=== WANDERING DETECTION ===")
    print(f"AUC={auc:.3f} F1={bestf1:.2f} precision={prec:.2f} recall={rec:.2f} accuracy={acc:.2f}")
    raw_pct,cwtr_pct = confgated_experiment()
    print("\n=== CONFIDENCE-GATED FALSE TRIGGER (purposeful walks w/ multipath) ===")
    print(f"raw-fix detector false-trigger    = {raw_pct:.0f}%")
    print(f"CWTR-gated detector false-trigger = {cwtr_pct:.0f}%")

if __name__ == "__main__":
    _print_ablation()
    _print_detector_sections()
    # ---------------- Behaviour-adaptive bandit cueing (illustrative) ----------------
    arms=np.array([[0.35,0.08],[0.25,0.16],[0.14,0.24],[0.10,0.30],[0.20,0.14]])
    trueJ=arms[:,0]+arms[:,1]; default=trueJ[0]; optimal=trueJ.min()
    r=np.random.default_rng(31); eps=0.1; realized=[]
    for user in range(150):
        Nc=np.zeros(5); Qm=np.zeros(5)
        for s in range(40):
            a=r.integers(5) if (r.random()<eps or np.all(Nc==0)) else int(np.argmin(np.where(Nc>0,Qm,np.inf)))
            err=1.0 if r.random()<arms[a,0] else 0.0; j=err+arms[a,1]; Nc[a]+=1; Qm[a]+=(j-Qm[a])/Nc[a]
        realized.append(trueJ[int(np.argmin(np.where(Nc>0,Qm,np.inf)))])
    realized=float(np.mean(realized))
    print("\n=== BEHAVIOUR-ADAPTIVE BANDIT CUEING (illustrative model) ===")
    print(f"fixed-default J={default:.2f}  adaptive policy J={realized:.2f}  optimal J={optimal:.2f}  reduction={100*(1-realized/default):.0f}%")

    # ---------------- Profile cost matrix (documented model) ----------------
    C=np.array([[0.80,0.25,0.55,0.45],[0.60,0.45,0.20,0.50],[0.55,0.50,0.60,0.22]])
    adaptive=C.min(axis=1).mean(); fixed=C[:,0].mean()
    print("\n=== PROFILE COST MATRIX (documented model) ===")
    print(f"adaptive={adaptive:.2f}  fixed-arrows={fixed:.2f}  reduction={100*(1-adaptive/fixed):.0f}%")

    # ---------------- Runtime ----------------
    r=np.random.default_rng(99); truth=S.gen_truth(r); z,a,sig=S.gen_obs(truth,S.PRIMARY_RATE,r)
    z=np.tile(z,(5,1))[:1000]
    t0=time.perf_counter()
    for _ in range(20): S.kalman_rts(z,[np.eye(2)*S.SIGMA0**2]*1000)
    print("\n=== RUNTIME ===")
    print(f"CWTR over 1000 fixes: {(time.perf_counter()-t0)/20*1000:.1f} ms (single-thread Python)")
