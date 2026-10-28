"""Conditional-on-fitted-model paired person-cluster uncertainty and same-fit SHAP."""
from __future__ import annotations
import argparse,itertools
from concurrent.futures import ProcessPoolExecutor
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import *
from feature_definitions import FEATURE_GROUPS

class FastScore:
    def __init__(self,y,p):
        self.y=np.asarray(y,int);self.p=np.asarray(p,float);self.order=np.argsort(-self.p,kind='stable')
        self.starts=np.r_[0,1+np.flatnonzero(np.diff(self.p[self.order])!=0)]
        self.bins=np.minimum((np.clip(self.p,0,1)*10).astype(int),9);self.err=(self.p-self.y)**2
    def score(self,w,cal=True):
        y=self.y;o=self.order;gy=np.add.reduceat((w*y)[o],self.starts);gw=np.add.reduceat(w[o],self.starts);gn=gw-gy
        pos=gy.sum();total=gw.sum();neg=total-pos;cy=np.cumsum(gy);cw=np.cumsum(gw)
        ap=np.sum(gy*np.divide(cy,cw,out=np.zeros_like(cy),where=cw>0))/pos
        auc=np.sum(gn*(cy-.5*gy))/(pos*neg)
        before=cw-gw;frac=np.divide(np.clip(.2*total-before,0,gw),gw,out=np.zeros_like(gw),where=gw>0);cap=np.sum(frac*gy)/pos
        ece=np.abs(np.bincount(self.bins,weights=w*(y-self.p),minlength=10)).sum()/total
        out={'roc_auc':auc,'pr_auc':ap,'brier':np.sum(w*self.err)/total,'recall20':cap,'lift20':cap/.2,'ece10':ece}
        if cal:
            a,s,_=calibration_coeff(y,self.p,w);out.update(calibration_intercept=a,calibration_slope=s)
        return out

def draws_chunk(args):
    frame,cols,start,stop,seed,cal=args;ids,code=np.unique(frame.pid,return_inverse=True);scores={c:FastScore(frame.exit_label_t1,frame[c]) for c in cols};out=[]
    for rep in range(start,stop):
        rng=np.random.default_rng(np.random.SeedSequence([seed,rep]));freq=rng.multinomial(len(ids),np.ones(len(ids))/len(ids));w=freq[code].astype(float)
        for c,s in scores.items():out.append({'replicate':rep,'model':c,**s.score(w,cal)})
    return pd.DataFrame(out)

def infer(frame,cols,nboot=1000,cal=True):
    args=[(frame[['pid','exit_label_t1']+cols],cols,i,min(i+100,nboot),SEED,cal) for i in range(0,nboot,100)]
    with ProcessPoolExecutor(max_workers=3) as ex:
        pieces=[]
        for x in ex.map(draws_chunk,args):pieces.append(x);print('BOOTSTRAP',sum(len(p)//len(cols) for p in pieces),'of',nboot,flush=True)
    draws=pd.concat(pieces,ignore_index=True);out=[]
    for c in cols:
        point=FastScore(frame.exit_label_t1,frame[c]).score(np.ones(len(frame)),cal)
        for metric in point:
            lo,hi=draws[draws.model.eq(c)][metric].quantile([.025,.975]);out.append({'model':c,'metric':metric,'estimate':point[metric],'lower':lo,'upper':hi,'resamples':nboot,'clusters':frame.pid.nunique(),'rows':len(frame),'bootstrap_unit':'person'})
    return pd.DataFrame(out),draws

def inference():
    t=pd.read_pickle(PRIV/'test_predictions.pkl');names=BASES+['forward_stack'];cols=[n+s for n in names for s in ['','_platt','_isotonic']]
    ci,draws=infer(t,cols,1000,True);save(ci,'primary_cluster_confidence_intervals.csv');draws.to_pickle(PRIV/'bootstrap_draws.pkl');save(draws,'aggregate_bootstrap_replicates.csv')
    pairs=list(itertools.combinations(names,2))+[('catboost','catboost_platt'),('catboost','catboost_isotonic'),('forward_stack','forward_stack_platt'),('catboost_platt','forward_stack_platt')];out=[]
    for a,b in pairs:
        aa=draws[draws.model.eq(a)].set_index('replicate');bb=draws[draws.model.eq(b)].set_index('replicate')
        for metric in ['roc_auc','pr_auc','brier','ece10']:
            delta=aa[metric]-bb[metric];lo,hi=delta.quantile([.025,.975]);estimate=float(ci[(ci.model==a)&(ci.metric==metric)].estimate.iloc[0]-ci[(ci.model==b)&(ci.metric==metric)].estimate.iloc[0]);out.append({'model_A':a,'model_B':b,'metric':metric,'difference_A_minus_B':estimate,'lower':lo,'upper':hi,'resamples':1000,'unit':'paired_person'})
    save(pd.DataFrame(out),'paired_cluster_model_contrasts.csv')
    waves=[]
    for wave,g in t.groupby('wave'):
        cc,_=infer(g,['catboost','catboost_platt','forward_stack','forward_stack_platt'],500,True);cc['wave']=wave;waves.append(cc)
    save(pd.concat(waves),'per_wave_cluster_confidence_intervals.csv')
    print('INFERENCE COMPLETE',flush=True)

def shap_analysis():
    d=load_data();t=d[d.wave.between(24,26)];f=joblib.load(PRIV/'models/catboost.joblib');cols=f['cols'];rng=np.random.default_rng(SEED);idx=rng.choice(len(t),size=3000,replace=False)
    pieces=[];groups=[];matrix=[]
    custom={'age_family':[c for c in cols if c.startswith('age_') or c in ['age_final','age_squared','older_worker_flag','young_worker_flag']],'job_context':[c for c in cols if c in ['industry_major','occupation_major','workplace_type','firm_size_band','worksite_size_band']]}
    for k in range(3):
        subset=t.iloc[idx[k*1000:(k+1)*1000]];X=clean_X(subset,cols);pool=Pool(X,cat_features=[c for c in cols if c not in NUMERIC_FEATURES]);sv=f['model'].get_feature_importance(pool,type='ShapValues',thread_count=3)[:,:-1]
        importance=np.mean(np.abs(sv),axis=0);pieces.append(pd.DataFrame({'sample':k+1,'feature':cols,'mean_abs_shap':importance}));matrix.append(importance)
        for group,features in {**FEATURE_GROUPS,**custom}.items():
            ix=[cols.index(c) for c in features if c in cols];groups.append({'sample':k+1,'group':group,'feature_n':len(ix),'mean_abs_sum_shap':float(np.mean(np.abs(sv[:,ix].sum(axis=1)))),'sum_mean_abs_shap':float(np.mean(np.abs(sv[:,ix]),axis=0).sum()),'group_set':'original_nonoverlapping_blocks' if group in FEATURE_GROUPS else 'overlapping_correlated_family_diagnostic'})
        print('SHAP SAMPLE',k+1,flush=True)
    df=pd.concat(pieces);save(df,'shap_sample_importance.csv');save(pd.DataFrame(groups),'shap_group_attributions.csv');save(df.groupby('feature').mean(numeric_only=True).drop(columns='sample').sort_values('mean_abs_shap',ascending=False).reset_index(),'shap_primary_importance.csv')
    stability=[]
    for i,j in itertools.combinations(range(3),2):
        topi=set(np.array(cols)[np.argsort(-matrix[i])[:20]]);topj=set(np.array(cols)[np.argsort(-matrix[j])[:20]]);stability.append({'sample_A':i+1,'sample_B':j+1,'rank_spearman':spearmanr(matrix[i],matrix[j]).statistic,'top20_overlap':len(topi&topj),'same_model_iterations':f['model'].tree_count_,'sample_rows':1000})
    save(pd.DataFrame(stability),'shap_rank_stability.csv')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['inference','shap']);args=a.parse_args();{'inference':inference,'shap':shap_analysis}[args.stage]()
