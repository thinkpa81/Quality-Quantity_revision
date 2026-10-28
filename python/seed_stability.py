"""Ten fixed seeds: full stochastic base refits and forward-OOF meta refits.

Deterministic logistic predictions are reused. Each stochastic fit retains the
primary feature rule, rows and hyperparameters. These descriptive distributions
measure algorithmic randomness conditional on this dataset, not sampling CIs.
"""
from __future__ import annotations
import argparse,gc
from concurrent.futures import ProcessPoolExecutor
import joblib
from common import *

FOLDS=[(7,9,11),(10,12,14),(13,15,17),(16,18,20)]
CALENDAR=['survey_year_centered','post_2018_workweek_reform','covid_period','post_covid_period']

def one_seed(seed):
    d=load_data();tr=d[d.wave.le(20)];va=d[d.wave.between(21,23)];te=d[d.wave.between(24,26)]
    cache=PRIV/'seed_scores';cache.mkdir(exist_ok=True)
    destination=cache/f'{seed}.pkl'
    if destination.exists():return pd.read_pickle(destination)['metrics']
    cols=feature_names(tr);oof=pd.read_pickle(PRIV/'forward_oof.pkl')
    v=pd.read_pickle(PRIV/'base_valid_predictions.pkl');t=pd.read_pickle(PRIV/'base_test_predictions.pkl')
    ov=oof[BASES].copy();vv=v[BASES].copy();tt=t[BASES].copy();rows=[];timings=[]
    cfg={'seed':seed,'threads':3}
    def scores(key,name,train,valid,test,features):
        p=cache/f'{seed}_{key}.npz'
        if p.exists():
            z=np.load(p);return z['valid'],z['test']
        fitted=fit_model(name,train,features,cfg);pv=predict(fitted,valid);pt=predict(fitted,test)
        timings.append({'seed':seed,'stage':key,'seconds':fitted['seconds'],'train_n':len(train),'features':len(features)})
        np.savez_compressed(p,valid=pv,test=pt);del fitted;gc.collect()
        print('SEED FIT',seed,key,flush=True);return pv,pt
    if seed!=SEED:
        for name in BASES[1:]:
            vv[name],tt[name]=scores('full_'+name,name,tr,va,te,cols)
        for end,start,stop in FOLDS:
            a=tr[tr.wave.le(end)];b=tr[tr.wave.between(start,stop)];fc=feature_names(a)
            assert a.wave.max()+1<b.wave.min()
            for name in BASES[1:]:
                pp,_=scores(f'fold{end}_'+name,name,a,b,b.iloc[:1],fc)
                ov.loc[b.index,name]=pp
    assert np.isfinite(ov.to_numpy()).all()
    meta=LogisticRegression(C=1,max_iter=2000).fit(ov,oof.exit_label_t1.astype(int))
    vv['forward_stack']=meta.predict_proba(vv[BASES])[:,1];tt['forward_stack']=meta.predict_proba(tt[BASES])[:,1]
    for name in ['catboost','forward_stack']:
        cal=calibrators(vv[name],va.exit_label_t1)
        for method in ['raw','platt','isotonic']:
            pv=calibrated(cal,vv[name],method);pt=calibrated(cal,tt[name],method)
            rows.append({'seed':seed,'model':name,'method':method,**metrics(te.exit_label_t1,pt,f1_threshold(va.exit_label_t1,pv))})
    # The same seed is paired across feature-removal fits; constant feature
    # removal can change implementation feature indexing and random draws.
    for label,drop in [('no_year',['survey_year_centered']),('no_calendar',CALENDAR),('no_constant_period_flags',CALENDAR[1:])]:
        pp,qq=scores(label,'catboost',tr,va,te,feature_names(tr,drop))
        cal=calibrators(pp,va.exit_label_t1)
        for method in ['raw','platt']:
            pv=calibrated(cal,pp,method);pt=calibrated(cal,qq,method)
            rows.append({'seed':seed,'model':label,'method':method,**metrics(te.exit_label_t1,pt,f1_threshold(va.exit_label_t1,pv))})
    out=pd.DataFrame(rows)
    payload={'metrics':out,'validation':vv,'test':tt,'forward_oof':ov,'meta_intercept':float(meta.intercept_[0]),'meta_coefficients':meta.coef_[0].tolist(),'timings':timings}
    pd.to_pickle(payload,destination);print('SEED COMPLETE',seed,flush=True);return out

def main(workers=2):
    seeds=list(range(SEED,SEED+10))
    with ProcessPoolExecutor(max_workers=workers) as pool:result=list(pool.map(one_seed,seeds))
    out=pd.concat(result,ignore_index=True);save(out,'seed_metrics.csv')
    contrasts=[]
    pairs=[('forward_stack','catboost'),('no_year','catboost'),('no_calendar','catboost'),('no_calendar','no_year'),('no_constant_period_flags','catboost')]
    for a,b in pairs:
        for method in ['raw','platt']:
            aa=out[(out.model==a)&(out.method==method)].set_index('seed');bb=out[(out.model==b)&(out.method==method)].set_index('seed')
            for metric in ['roc_auc','pr_auc','brier','calibration_intercept','calibration_slope']:
                delta=aa[metric]-bb[metric]
                contrasts.append({'model_A':a,'model_B':b,'method':method,'metric':metric,'seeds':len(delta),'mean_A_minus_B':delta.mean(),'sd_A_minus_B':delta.std(ddof=1),'minimum':delta.min(),'maximum':delta.max(),'positive_differences':int((delta>0).sum()),'negative_differences':int((delta<0).sum())})
    save(pd.DataFrame(contrasts),'seed_contrasts.csv')
    summary=out.groupby(['model','method'])[['roc_auc','pr_auc','brier','calibration_intercept','calibration_slope']].agg(['mean','std','min','max']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns];save(summary,'seed_summary.csv')
    dump({'seeds':seeds,'model_threads':3,'same_threads_as_primary':True,'stochastic_models':['random_forest','xgboost','catboost'],'reused_deterministic_model':'logistic','refitted_stages':['full training','four forward OOF blocks','meta learner','validation calibration and thresholds'],'interpretation':'Descriptive optimization randomness conditional on fixed data and development protocol; not confidence intervals for sampling or protocol-selection uncertainty'},RES/'seed_design.json')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);args=ap.parse_args();main(args.workers)
