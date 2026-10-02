"""Participant-balanced drug metrics. R2 is weighted across queries, not averaged per subject."""
import numpy as np

def balanced_metrics(y, pred, participants):
    y=np.asarray(y,dtype=float);pred=np.asarray(pred,dtype=float)
    if not (np.isfinite(y).all() and np.isfinite(pred).all()):raise ValueError('Nonfinite log concentration')
    _,codes,counts=np.unique(participants,return_inverse=True,return_counts=True)
    w=1/(len(counts)*counts[codes]);e=pred-y
    c=np.exp(y);chat=np.exp(pred)
    if not (np.isfinite(c).all() and np.isfinite(chat).all()):raise ValueError('Nonfinite concentration')
    def r2(a,b):
        den=np.sum(w*(a-np.sum(w*a))**2)
        if len(a)<2 or den<=np.finfo(float).eps*max(np.sum(w*a*a),np.finfo(float).tiny):return np.nan
        return 1-np.sum(w*(a-b)**2)/den
    return dict(log_rmse=np.sqrt(np.sum(w*e**2)),log_mae=np.sum(w*abs(e)),within_twofold=100*np.sum(w*(abs(e)<=np.log(2))),signed_log_bias=np.sum(w*e),linear_rmse=np.sqrt(np.sum(w*(chat-c)**2)),linear_mae=np.sum(w*abs(chat-c)),r2_log=r2(y,pred),r2_linear=r2(c,chat),participants=len(counts),queries=len(y))
