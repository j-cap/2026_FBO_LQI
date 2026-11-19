"""Instrumented single-episode replay with explicitly supplied controller gains."""
import numpy as np
from src.actuator_bandwidth import augment_discrete,design,step
from src.actuator_antiwindup import integral_update


def gains(ac,bc,tau,xi,cc,ts):
    ad,bd=augment_discrete(ac,bc,tau,ts)
    k,kr,radius=design(ad,bd,xi,cc['state_scale'],ts,cc['fixed_R'])
    return dict(K=k.tolist(),kr=kr,design_radius=radius)


def episode(client,ac,bc,tau,controller_gains,bank,ep,cc,ts,substeps=5,trace=False):
    """Record true and noisy error on one unchanged noisy trajectory.

    K and kr are supplied; no redesign takes place inside this routine. A target
    linear radius >=1 is reported, not used to censor a finite nonlinear rollout.
    """
    K=np.asarray(controller_gains['K']);kr=controller_gains['kr']
    ad,bd=augment_discrete(ac,bc,tau,ts)
    ai=np.block([[ad,np.zeros((3,1))],[ts*np.array([[0.,1.,0.]]),np.array([[.998]])]])
    bi=np.vstack([bd,[[0.]]]);radius=float(max(abs(np.linalg.eigvals(ai-bi@K[None,:]))))
    _,ref=bank.refs[ep.T_lc];rng=np.random.RandomState(ep.seed)
    x=np.zeros(3);eta=kr*float(ref[0])/K[3]
    measured=[];true=[];noises=[];cross=[];rows=[];max_beta=max_delta=max_rate=0.;sat=0
    reason=''
    for i,rr in enumerate(ref[:-1]):
        raw=-K[:3]@x-K[3]*eta+kr*rr
        u=float(np.clip(raw,cc['u_min'],cc['u_max']))
        noise=cc['noise_std']*rng.randn();y=x[1]+noise;err=x[1]-rr
        measured.append((y-rr)**2);true.append(err**2);noises.append(noise**2);cross.append(2*err*noise)
        if trace:rows.append(dict(time=i*ts,beta=float(x[0]),yaw=float(x[1]),reference=float(rr),
            measured_yaw=float(y),delta=float(x[2]),raw=float(raw),command=u,eta=float(eta)))
        eta=integral_update(eta,y-rr,raw,u,K[3],ts)
        sat+=abs(u-raw)>1e-10;max_rate=max(max_rate,abs((u-x[2])/tau))
        for j in range(substeps):
            x=step(x,u,client.params,tau,ts/substeps,cc.get('tire_mu',1.))
            max_beta=max(max_beta,abs(x[0]));max_delta=max(max_delta,abs(x[2]))
            if not np.isfinite(x).all() or max_beta>np.deg2rad(cc['beta_max_deg']):
                reason='beta_limit';break
        if reason:break
        x[:2]+=rng.multivariate_normal(np.zeros(2),np.diag(cc['process_noise']))
    if not reason and (max_rate>cc['u_rate_max']/ts or max_delta>max(abs(cc['u_min']),abs(cc['u_max']))+1e-8):
        reason='actuator_limit'
    complete=len(measured)==len(ref)-1 and reason!='beta_limit'
    # Do not report a truncated trajectory's RMSE as a full episode metric.
    mean=lambda values:float(np.mean(values)) if complete else None
    result=dict(cost=float(np.sqrt(np.mean(measured))) if complete else float('nan'),feasible=not bool(reason),reason=reason,
        max_beta=float(max_beta),max_delta=float(max_delta),max_rate=float(max_rate),saturation_fraction=float(sat/len(measured)),
        spectral_radius=radius,true_cost=float(np.sqrt(np.mean(true))) if complete else None,
        measured_mse=mean(measured),true_mse=mean(true),noise_mse=mean(noises),cross_mse=mean(cross),
        samples=len(measured),complete=complete)
    if trace:result['trace']=rows
    return result
