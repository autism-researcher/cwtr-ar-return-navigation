import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cwtr_simulation as S

OUT="../figures"; import os; os.makedirs(OUT,exist_ok=True)
RATE=S.PRIMARY_RATE
# High-quality output: vector PDF (used by LaTeX) + 300-dpi PNG fallback.
# pdf.fonttype=42 embeds TrueType fonts so text stays selectable/sharp.
plt.rcParams.update({'font.size':10,'savefig.bbox':'tight','savefig.dpi':300,
                     'figure.dpi':150,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':0.8})

def save(name):
    plt.savefig(f"{OUT}/{name}.pdf")           # vector, primary
    plt.savefig(f"{OUT}/{name}.png", dpi=300)   # high-res raster fallback
    plt.close()

# ---- light, colorblind-friendly palette (kept legible in grayscale via
#      distinct line styles, markers, dark edges, and bar hatching) ----
GRAY1='0.72'; GRAY2='0.82'          # baselines (light grays)
BLUE  ='#9DC3E6'                    # light blue  -> Kalman
RED   ='#E69B96'                    # light salmon -> CWTR (hero)
GREEN ='#B5D6A0'                    # light green
ORANGE='#F4C28A'                    # light orange
EDGE  ='0.30'                       # dark thin edge so light fills read on white

# ---- one representative trial (truth, raw, kalman, cwtr) ----
r=np.random.default_rng(3)
truth=S.gen_truth(r); z,a,sig=S.gen_obs(truth,RATE,r)
mv=np.mean(sig**2); kal=S.kalman_rts(z,[np.eye(2)*mv]*len(z))
c=S.conf(z,a); Rc=[np.eye(2)*((S.SIGMA0**2/c[t])*(S.GATE if c[t]<S.TAU else 1.0)) for t in range(len(z))]
cwtr=S.kalman_rts(z,Rc)
plt.figure(figsize=(5,4.2))
plt.plot(z[:,0],z[:,1],'.',color=GRAY1,ms=4,label='Raw fixes (multipath)')
plt.plot(truth[:,0],truth[:,1],'-',color='0.35',lw=1.6,label='Ground truth')
plt.plot(kal[:,0],kal[:,1],'--',color=BLUE,lw=1.8,label='Kalman/RTS (fixed R)')
plt.plot(cwtr[:,0],cwtr[:,1],'-',color=RED,lw=2.2,label='CWTR (proposed)')
plt.plot(truth[0,0],truth[0,1],'o',color=GREEN,mec=EDGE,mew=1.0,ms=10,label='Safe origin')
plt.axis('equal'); plt.legend(fontsize=7.5,loc='best'); plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.title('Single simulated trial (heavy multipath)')
save("fig2_singletrial")

# ---- RMSE bar chart at primary point ----
out=S.run(RATE)
labels=['Raw','Decim.\n(5 m)','Acc.+\ndecim.','Kalman/\nRTS','$\\chi^2$-gated\nKF','CWTR']
keys=['raw','dec','accdec','kalman','chikf','cwtr']
means=[out[k][:,0].mean() for k in keys]; sds=[out[k][:,0].std() for k in keys]
cols=[GRAY1,GRAY1,GRAY2,BLUE,GREEN,RED]
hatches=['//','\\\\','xx','..','oo',None]
plt.figure(figsize=(5.4,3.6))
bars=plt.bar(range(len(keys)),means,yerr=sds,capsize=4,color=cols,edgecolor=EDGE,linewidth=0.8)
for b,h in zip(bars,hatches):
    if h: b.set_hatch(h)
plt.xticks(range(len(keys)),labels,fontsize=8); plt.ylabel('Position RMSE (m)')
for i,m in enumerate(means): plt.text(i,m+sds[i]+0.3,f"{m:.2f}",ha='center',fontsize=8)
plt.title('Reconstruction RMSE (M=120, heavy multipath)')
save("fig3_rmse")

# ---- ablation ----
def cwtr_rmse(rate,use_acc,use_kin,seed=S.SEED):
    rr=np.random.default_rng(seed); o=[]
    for _ in range(S.M):
        tr=S.gen_truth(rr); zz,aa,ss=S.gen_obs(tr,rate,rr); n=len(zz)
        cc=S.conf(zz,aa,use_acc=use_acc,use_kin=use_kin)   # canonical confidence
        R=[np.eye(2)*((S.SIGMA0**2/cc[t])*(S.GATE if cc[t]<S.TAU else 1.0)) for t in range(n)]
        o.append(S.metrics(S.kalman_rts(zz,R),tr)[0])
    return np.mean(o)
ab=[cwtr_rmse(RATE,True,True),cwtr_rmse(RATE,True,False),cwtr_rmse(RATE,False,True)]
plt.figure(figsize=(4.6,3.4))
abar=plt.bar(['Full CWTR','No kinematic\nfactor','No accuracy\nfactor'],ab,color=[RED,ORANGE,GREEN],edgecolor=EDGE,linewidth=0.8)
for b,h in zip(abar,[None,'//','xx']):
    if h: b.set_hatch(h)
for i,m in enumerate(ab): plt.text(i,m+0.05,f"{m:.2f}",ha='center',fontsize=9)
plt.ylabel('RMSE (m)'); plt.title('Confidence-component ablation')
save("fig4_ablation")

# ---- multipath sensitivity sweep (crossover) ----
rates=[0.02,0.04,0.06,0.08,0.12,0.18]; raw=[];kal_=[];chi_=[];cw=[]
for rt in rates:
    o=S.run(rt); raw.append(o['raw'][:,0].mean()); kal_.append(o['kalman'][:,0].mean())
    chi_.append(o['chikf'][:,0].mean()); cw.append(o['cwtr'][:,0].mean())
plt.figure(figsize=(5,3.6))
plt.plot(rates,raw,'s-',color=GRAY1,mec=EDGE,mew=0.8,lw=1.8,label='Raw fixes')
plt.plot(rates,kal_,'^--',color=BLUE,mec=EDGE,mew=0.8,lw=1.8,label='Kalman/RTS (fixed R)')
plt.plot(rates,chi_,'d-.',color=GREEN,mec=EDGE,mew=0.8,lw=1.8,label='$\\chi^2$-gated KF (adapt. R)')
plt.plot(rates,cw,'o-',color=RED,mec=EDGE,mew=0.8,lw=2.2,label='CWTR (proposed)')
plt.axvline(0.06,color='0.8',ls=':'); plt.text(0.061,raw[0],'crossover',fontsize=7.5,color='0.4')
plt.xlabel('Multipath outlier rate'); plt.ylabel('RMSE (m)'); plt.legend(fontsize=8)
plt.title('Sensitivity to multipath severity')
save("fig5_noise")

# ---- wandering score distributions (Eq. (6) score; AUC computed live) ----
from extra_modules import wandering_experiment, confgated_experiment
_auc,_f1,_prec,_rec,_acc,sp,sw = wandering_experiment()
plt.figure(figsize=(5,3.4))
plt.hist(sp,bins=30,alpha=0.75,color=GREEN,edgecolor=EDGE,linewidth=0.5,label='Purposeful')
plt.hist(sw,bins=30,alpha=0.75,color=RED,edgecolor=EDGE,linewidth=0.5,hatch='//',label='Wandering')
plt.xlabel('Wandering irregularity score $S_W$'); plt.ylabel('Count'); plt.legend(fontsize=8)
plt.title(f'Wandering detection (AUC = {_auc:.2f}, computed)')
save("fig7_wandering")

# ---- bandit learning curve (illustrative model) ----
arms=np.array([[0.35,0.08],[0.25,0.16],[0.14,0.24],[0.10,0.30],[0.20,0.14]])
trueJ=arms[:,0]+arms[:,1]; default=trueJ[0]; optimal=trueJ.min()
r=np.random.default_rng(31); eps=0.1; curve=np.zeros(40)
for user in range(150):
    N=np.zeros(5);Qm=np.zeros(5)
    for s in range(40):
        a=r.integers(5) if (r.random()<eps or np.all(N==0)) else int(np.argmin(np.where(N>0,Qm,np.inf)))
        err=1.0 if r.random()<arms[a,0] else 0.0;j=err+arms[a,1];N[a]+=1;Qm[a]+=(j-Qm[a])/N[a]
        fa=int(np.argmin(np.where(N>0,Qm,np.inf))); curve[s]+=trueJ[fa]
curve/=150
plt.figure(figsize=(5,3.4))
plt.plot(range(1,41),curve,'-',color=RED,lw=2.2,label='Adaptive policy (true J)')
plt.axhline(default,ls='--',color=GRAY1,lw=1.6,label=f'Fixed default ({default:.2f})')
plt.axhline(optimal,ls=':',color=GREEN,lw=1.8,label=f'Model optimum ({optimal:.2f})')
plt.xlabel('Session'); plt.ylabel('Error + load objective J'); plt.legend(fontsize=8)
plt.title('Behaviour-adaptive cueing (illustrative model)')
save("fig8_bandit")

# ---- profile cost matrix (light grayscale) ----
C=np.array([[0.80,0.25,0.55,0.45],[0.60,0.45,0.20,0.50],[0.55,0.50,0.60,0.22]])
prof=['ASD','Child','Elderly']; cue=['Arrows','Footprints','Cartoon','Panels']
plt.figure(figsize=(4.8,3.2))
im=plt.imshow(C,cmap='Greys',aspect='auto',vmin=0,vmax=1.4)  # light grayscale
plt.colorbar(im,label='Modeled cognitive cost')
plt.xticks(range(4),cue); plt.yticks(range(3),prof)
for i in range(3):
    for j in range(4):
        plt.text(j,i,f"{C[i,j]:.2f}",ha='center',va='center',
                 color='black',fontsize=9,
                 fontweight='bold' if C[i,j]==C[i].min() else 'normal')
plt.title('Modeled profile cost matrix (adaptive -66%)')
save("fig9_costmatrix")

# ---- confidence-gated false trigger (rates computed live) ----
_raw_pct,_cwtr_pct = confgated_experiment()
plt.figure(figsize=(4.2,3.4))
gbar=plt.bar(['Raw-fix\ndetector','CWTR-gated\ndetector'],[_raw_pct,_cwtr_pct],color=[RED,GREEN],edgecolor=EDGE,linewidth=0.8)
gbar[0].set_hatch('//')
plt.text(0,_raw_pct+1,f'{_raw_pct:.0f}%',ha='center'); plt.text(1,_cwtr_pct+1,f'{_cwtr_pct:.0f}%',ha='center')
plt.ylabel('False-trigger rate on purposeful motion (%)'); plt.ylim(0,110)
plt.title('Confidence-gated disorientation detection')
save("fig10_confgated")
print("figures written (pdf + png):", sorted(os.listdir(OUT)))
