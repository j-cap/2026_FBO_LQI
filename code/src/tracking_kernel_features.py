"""Cost-label-independent feature bases on the declared public action grid.

All bases retain the exact constant feature. Only the Matern signal kernel is
approximated. These factors can use D's unchanged Gaussian message algebra.
"""
import time

import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.special import logsumexp

from .tracking_budget import matern
from .tracking_online import features


def pivoted_factor(kernel,rank):
    """Greedy maximum-residual-diagonal pivots; no labels or client histories."""
    n=len(kernel)
    if not 1 <= rank <= n:
        raise ValueError('Invalid pivoted-Cholesky rank.')
    factor=np.zeros((n,rank))
    residual=np.diag(kernel).copy()
    pivots=[]
    for j in range(rank):
        pivot=int(np.argmax(residual))
        if residual[pivot] <= 1e-13:
            return factor[:,:j],pivots
        column=kernel[:,pivot]-factor[:,:j]@factor[pivot,:j]
        factor[:,j]=column/np.sqrt(residual[pivot])
        residual=np.maximum(0.,residual-factor[:,j]**2)
        pivots.append(pivot)
    return factor,pivots


def basis_bank(domain,gp,model,specifications):
    """Return prior-whitened public features F, so the base covariance is F F^T."""
    signal=matern(domain.X,domain.X,gp['length_scales'],gp['amplitude'],0.)
    exact=signal+gp['constant_variance']
    eigen=None
    result={}
    for item in specifications:
        started=time.perf_counter()
        family,rank=item['family'],item['rank']
        name=f'{family}_{rank}'
        if name in result:
            raise ValueError('Duplicate kernel representation.')
        pivots=[]
        if family == 'rff':
            phi=features(domain,dict(model,features=rank))
            F=phi*np.r_[np.sqrt(gp['constant_variance']),np.full(rank,gp['amplitude'])]
        else:
            if not 1 <= rank <= len(signal):
                raise ValueError('Invalid deterministic basis rank.')
            if family == 'spectral':
                if eigen is None:
                    values,vectors=np.linalg.eigh(signal)
                    values,vectors=values[::-1],vectors[:,::-1]
                    signs=np.sign(vectors[np.argmax(np.abs(vectors),axis=0),np.arange(len(values))])
                    eigen=(np.maximum(values,0.),vectors*np.where(signs == 0,1,signs))
                values,vectors=eigen
                part=vectors[:,:rank]*np.sqrt(values[:rank])
            elif family == 'pivoted_cholesky':
                part,pivots=pivoted_factor(signal,rank)
            else:
                raise ValueError('Unknown public feature family.')
            F=np.column_stack((np.full(len(signal),np.sqrt(gp['constant_variance'])),part))
        kernel=F@F.T
        result[name]=dict(family=family,rank=rank,F=F,kernel=kernel,pivots=pivots,
            feature_width=F.shape[1],construction_seconds=time.perf_counter()-started,
            relative_kernel_Frobenius_error=float(np.linalg.norm(kernel-exact)/np.linalg.norm(exact)),
            max_relative_diagonal_error=float(np.max(np.abs(np.diag(kernel-exact))/np.diag(exact))),
            public_basis_bytes=int(F.nbytes))
    return exact,result


def query_posterior(histories,kernel,hierarchical,gp,query_clients,query_actions):
    """D's unchanged covariance model, evaluated in observation space for efficiency.

    This is equivalent to feature-space Gaussian inference when kernel=F F^T.
    Queries carry client/input identities only, never outcomes.
    """
    observations=[(i,o['action'],np.log(o['J'])) for i,h in enumerate(histories) for o in h if o['eligible']]
    tasks=np.asarray([i for i,_,_ in observations],int)
    actions=[a for _,a,_ in observations]
    y=np.asarray([v for _,_,v in observations])
    query_clients,query_actions=np.asarray(query_clients,int),np.asarray(query_actions,int)
    if query_clients.shape != query_actions.shape:
        raise ValueError('Query client/input shapes differ.')
    means,variances,evidence=[],[],[]
    for rho in hierarchical['sharing_fractions']:
        if not len(y):
            means.append(np.zeros(len(query_actions)));variances.append(np.diag(kernel)[query_actions]);evidence.append(0.)
            continue
        K=kernel[np.ix_(actions,actions)]*(rho+(1-rho)*(tasks[:,None] == tasks))
        K.flat[::len(K)+1] += gp['noise_std_log_J']**2+1e-9
        factor=cho_factor(K,lower=True,check_finite=False)
        alpha=cho_solve(factor,y,check_finite=False)
        evidence.append(float(-.5*(y@alpha+2*np.log(np.diag(factor[0])).sum()+len(y)*np.log(2*np.pi))))
        cross=kernel[np.ix_(query_actions,actions)]*(rho+(1-rho)*(query_clients[:,None] == tasks))
        means.append(cross@alpha)
        variance=np.diag(kernel)[query_actions]-np.einsum('ij,ji->i',cross,
            cho_solve(factor,cross.T,check_finite=False))
        if np.min(variance,initial=0.) < -1e-8:
            raise RuntimeError('Negative kernel posterior variance.')
        variances.append(np.maximum(variance,0.))
    prior=np.asarray(hierarchical['prior_weights'],float)
    log_weights=np.log(prior/prior.sum())+evidence
    weights=np.exp(log_weights-logsumexp(log_weights))
    means,variances=np.asarray(means),np.asarray(variances)
    mean=np.einsum('k,kq->q',weights,means)
    variance=np.einsum('k,kq->q',weights,variances+(means-mean[None])**2)
    return dict(mean=mean,variance=variance,sharing_weights=weights,
        expected_sharing_fraction=float(weights@hierarchical['sharing_fractions']))
