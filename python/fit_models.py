"""Fixed primary refits and temporally ordered out-of-fold stacking."""
from __future__ import annotations
import json,time,itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from common import *

def main():
    d=load_data();tr=d[d.wave.le(20)];va=d[d.wave.between(21,23)];te=d[d.wave.between(24,26)]
    modeldir=PRIV/'models';modeldir.mkdir(exist_ok=True)
    cols=feature_names(tr);dump(cols,RES/'model_features.json')
    metadata=['pid','wave','exit_label_t1','age_final','gender','region','cohort','destination','weight18_cross','weight18_long','hhid']
    v=va[metadata].copy();t=te[metadata].copy();runtimes=[]
    for name in BASES:
        path=modeldir/(name+'.joblib')
        if path.exists(): fitted=joblib.load(path)
        else:fitted=fit_model(name,tr,cols);joblib.dump(fitted,path)
        v[name]=predict(fitted,va);t[name]=predict(fitted,te)
        runtimes.append({'analysis':'primary','model':name,'train_n':len(tr),'features':len(cols),'seconds':fitted['seconds'],'training_weights':'none'})
        print('PRIMARY',name,metrics(te.exit_label_t1,t[name],cal=False),flush=True)
    v.to_pickle(PRIV/'base_valid_predictions.pkl');t.to_pickle(PRIV/'base_test_predictions.pkl')
    oof_path=PRIV/'forward_oof.pkl'
    if oof_path.exists():oof=pd.read_pickle(oof_path)
    else:
        pieces=[];foldrows=[]
        # Labels of each training row must be observed before the earliest OOF scoring wave.
        for fend,start,end in [(7,9,11),(10,12,14),(13,15,17),(16,18,20)]:
            ftr=tr[tr.wave.le(fend)];fo=tr[tr.wave.between(start,end)];fc=feature_names(ftr)
            out=fo[['pid','wave','exit_label_t1']].copy()
            assert ftr.wave.max()+1 < fo.wave.min()
            for name in BASES:
                path=modeldir/f'forward_{fend}_{name}.joblib'
                if path.exists():f=joblib.load(path)
                else:f=fit_model(name,ftr,fc);joblib.dump(f,path)
                out[name]=predict(f,fo)
                runtimes.append({'analysis':f'forward_{fend}_{start}_{end}','model':name,'train_n':len(ftr),'features':len(fc),'seconds':f['seconds'],'training_weights':'none'})
            pieces.append(out);foldrows.append({'train_start':1,'train_end':fend,'latest_training_outcome_wave':fend+1,'oof_start':start,'oof_end':end,'train_n':len(ftr),'oof_n':len(fo),'feature_n':len(fc)})
            print('FORWARD FOLD',fend,start,end,flush=True)
        oof=pd.concat(pieces);oof.to_pickle(oof_path);save(pd.DataFrame(foldrows),'forward_folds.csv')
    y=oof.exit_label_t1.astype(int)
    for name,weight in [('forward_stack',None),('forward_stack_balanced','balanced')]:
        meta=LogisticRegression(C=1,max_iter=2000,class_weight=weight).fit(oof[BASES],y)
        v[name]=meta.predict_proba(v[BASES])[:,1];t[name]=meta.predict_proba(t[BASES])[:,1]
        joblib.dump(meta,modeldir/(name+'.joblib'))
        save(pd.DataFrame({'base_model':BASES,'coefficient':meta.coef_[0],'intercept':meta.intercept_[0],'class_weight':str(weight)}),name+'_coefficients.csv')
    # Matched four-fold shuffled OOF sensitivity. Full-data refits are reused at prediction time.
    shuffled_path=PRIV/'shuffled_oof.pkl'
    if shuffled_path.exists():shuf=pd.read_pickle(shuffled_path)
    else:
        shuf=tr[['pid','wave','exit_label_t1']].copy();
        for name in BASES:shuf[name]=np.nan
        cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=SEED)
        for k,(fi,oi) in enumerate(cv.split(tr,tr.exit_label_t1)):
            ftr=tr.iloc[fi];fo=tr.iloc[oi];fc=feature_names(ftr)
            for name in BASES:
                path=modeldir/f'shuffled_{k}_{name}.joblib'
                if path.exists():f=joblib.load(path)
                else:f=fit_model(name,ftr,fc);joblib.dump(f,path)
                shuf.loc[fo.index,name]=predict(f,fo)
            print('SHUFFLED SENSITIVITY FOLD',k,flush=True)
        shuf.to_pickle(shuffled_path)
    # Match meta-training rows to the forward OOF cohort; base-training availability still differs.
    shuf_meta=shuf.loc[oof.index]
    meta=LogisticRegression(C=1,max_iter=2000).fit(shuf_meta[BASES],shuf_meta.exit_label_t1.astype(int))
    v['shuffled_stack']=meta.predict_proba(v[BASES])[:,1];t['shuffled_stack']=meta.predict_proba(t[BASES])[:,1]
    models=BASES+['forward_stack','forward_stack_balanced','shuffled_stack'];rows=[];thresholds=[]
    for name in models:
        calibrate=calibrators(v[name].to_numpy(),v.exit_label_t1.astype(int).to_numpy())
        joblib.dump(calibrate,modeldir/(name+'_calibrators.joblib'))
        for method in ['raw','platt','isotonic']:
            col=name if method=='raw' else name+'_'+method
            v[col]=calibrated(calibrate,v[name].to_numpy(),method);t[col]=calibrated(calibrate,t[name].to_numpy(),method)
            th=f1_threshold(v.exit_label_t1,v[col]);thresholds.append({'model':name,'method':method,'threshold':th})
            for part,frame in [('validation',v),('test',t)]:rows.append({'model':name,'method':method,'partition':part,**metrics(frame.exit_label_t1,frame[col],th)})
    v.to_pickle(PRIV/'valid_predictions.pkl');t.to_pickle(PRIV/'test_predictions.pkl')
    save(pd.DataFrame(rows),'model_calibration_metrics.csv');save(pd.DataFrame(thresholds),'selected_thresholds.csv');save(pd.DataFrame(runtimes),'training_runtimes.csv')
    print('PRIMARY MODELLING COMPLETE',flush=True)

if __name__=='__main__':main()
