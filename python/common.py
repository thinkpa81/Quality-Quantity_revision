from __future__ import annotations
import json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,precision_recall_curve
from catboost import CatBoostClassifier,Pool
from xgboost import XGBClassifier
from feature_definitions import MODEL_FEATURES,NUMERIC_FEATURES

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';PRIV=ROOT/'private'
SEED=20261026
BASES=['logistic','random_forest','xgboost','catboost']

def dump(obj,p):Path(p).write_text(json.dumps(obj,indent=2,default=lambda o:float(o) if isinstance(o,np.generic) else str(o)))
def save(df,name):df.to_csv(RES/name,index=False,encoding='utf-8-sig')
def load_data():return pd.read_pickle(PRIV/'analysis.pkl')
def feature_names(train,drop=()):return [c for c in MODEL_FEATURES if c in train and c not in drop and train[c].notna().sum()>=20]
def clean_X(df,cols):
    out=df[cols].copy()
    for c in cols:
        if c in NUMERIC_FEATURES:out[c]=pd.to_numeric(out[c],errors='coerce').astype(float).replace([np.inf,-np.inf],np.nan)
        else:out[c]=out[c].map(lambda x:'__MISSING__' if pd.isna(x) else str(x)).astype(object)
    return out
def fit_model(name,train,cols=None,config=None,target='exit_label_t1',weights=None):
    cols=feature_names(train) if cols is None else cols; cfg=config or {};X=clean_X(train,cols);y=train[target].to_numpy(dtype=int)
    cat=[c for c in cols if c not in NUMERIC_FEATURES];num=[c for c in cols if c in NUMERIC_FEATURES]
    t=time.time(); seed=cfg.get('seed',SEED); threads=cfg.get('threads',3)
    if name=='catboost':
        model=CatBoostClassifier(iterations=cfg.get('iterations',300),depth=cfg.get('depth',6),learning_rate=cfg.get('learning_rate',.05),loss_function='Logloss',random_seed=seed,thread_count=threads,verbose=False,allow_writing_files=False,l2_leaf_reg=cfg.get('l2_leaf_reg',3))
        model.fit(X,y,cat_features=cat,sample_weight=weights)
    else:
        pre=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),num),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=True),cat)],sparse_threshold=1.0)
        if name=='logistic':est=LogisticRegression(C=cfg.get('C',1.0),solver='lbfgs',max_iter=cfg.get('max_iter',2000),tol=1e-5,class_weight=cfg.get('class_weight',None))
        elif name=='random_forest':est=RandomForestClassifier(n_estimators=300,min_samples_leaf=5,max_depth=None,n_jobs=threads,random_state=seed,class_weight=cfg.get('class_weight',None))
        elif name=='xgboost':est=XGBClassifier(n_estimators=cfg.get('iterations',400),max_depth=cfg.get('depth',6),learning_rate=cfg.get('learning_rate',.05),subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=threads,random_state=seed)
        else:raise ValueError(name)
        model=Pipeline([('preprocess',pre),('estimator',est)])
        kw={'estimator__sample_weight':weights} if weights is not None else {}
        model.fit(X,y,**kw)
    return {'model':model,'cols':cols,'name':name,'seconds':time.time()-t,'config':cfg}
def predict(fitted,df):return fitted['model'].predict_proba(clean_X(df,fitted['cols']))[:,1]
def calibrators(p,y):
    p=np.asarray(p,float);y=np.asarray(y,int)
    z=logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1)
    # Near-unpenalized logistic recalibration on log odds; fitted on validation only.
    pl=LogisticRegression(C=1e8,max_iter=2000,tol=1e-9).fit(z,y)
    iso=IsotonicRegression(out_of_bounds='clip').fit(p,y)
    return {'platt':pl,'isotonic':iso}
def calibrated(cal,p,method):
    p=np.asarray(p,float)
    if method=='raw':return np.asarray(p)
    if method=='platt':return cal[method].predict_proba(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1))[:,1]
    return cal[method].predict(p)
def f1_threshold(y,p):
    precision,recall,th=precision_recall_curve(y,p)
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    return float(th[np.argmax(f)])
def calibration_coeff(y,p,w=None):
    y=np.asarray(y,float);p=np.clip(np.asarray(p,float),1e-6,1-1e-6);w=np.ones(len(y)) if w is None else np.asarray(w,float)
    z=logit(p);a=0.
    for _ in range(40):
        mu=expit(z+a);step=np.sum(w*(y-mu))/max(np.sum(w*mu*(1-mu)),1e-10);step=np.clip(step,-5,5);a+=step
        if abs(step)<1e-8:break
    X=np.column_stack([np.ones(len(z)),z]);b=np.array([0.,1.])
    for _ in range(40):
        mu=expit(X@b);H=X.T@((w*mu*(1-mu))[:,None]*X);score=X.T@(w*(y-mu))
        try:step=np.linalg.solve(H+np.eye(2)*1e-9,score)
        except np.linalg.LinAlgError:break
        step=np.clip(step,-3,3);b+=step
        if np.max(np.abs(step))<1e-8:break
    return float(a),float(b[1]),float(b[0])
def metrics(y,p,threshold=.5,weights=None,cal=True):
    y=np.asarray(y,int);p=np.asarray(p,float);w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    total=w.sum();events=np.sum(w*y);pred=p>=threshold
    tp=np.sum(w*y*pred);fp=np.sum(w*(1-y)*pred);fn=events-tp
    order=np.argsort(-p,kind='stable'); starts=np.r_[0,1+np.flatnonzero(np.diff(p[order])!=0)];gw=np.add.reduceat(w[order],starts);gy=np.add.reduceat((w*y)[order],starts);before=np.cumsum(gw)-gw;fraction=np.divide(np.clip(.2*total-before,0,gw),gw,out=np.zeros_like(gw),where=gw>0);capture=np.sum(fraction*gy)/events if events>0 else np.nan
    bins=np.minimum((np.clip(p,0,1)*10).astype(int),9)
    ece=sum(abs(np.sum(w[bins==b]*(y[bins==b]-p[bins==b]))) for b in range(10))/total
    result={'n':len(y),'events':int(y.sum()),'event_rate':events/total,'roc_auc':roc_auc_score(y,p,sample_weight=w) if events>0 and events<total else np.nan,'pr_auc':average_precision_score(y,p,sample_weight=w) if events>0 else np.nan,'brier':np.sum(w*(p-y)**2)/total,'recall20':capture,'lift20':capture/.2,'ece10':ece,'mean_probability':np.sum(w*p)/total,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/events if events else np.nan,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'positive_rate':np.sum(w*pred)/total,'threshold':threshold}
    if cal:
        a,s,b=calibration_coeff(y,p,w);result.update(calibration_intercept=a,calibration_slope=s,calibration_joint_intercept=b)
    return result

def cluster_bootstrap(frame,prob_cols,n_boot=1000,seed=SEED,calibration=False):
    rng=np.random.default_rng(seed);y=frame.exit_label_t1.to_numpy(int);ids,code=np.unique(frame.pid,return_inverse=True);samples=[]
    for b in range(n_boot):
        freq=rng.multinomial(len(ids),np.ones(len(ids))/len(ids));w=freq[code]
        for name in prob_cols:
            m=metrics(y,frame[name],weights=w,cal=calibration)
            samples.append({'replicate':b,'model':name,**{k:m[k] for k in ['roc_auc','pr_auc','brier','recall20','lift20','ece10']+(['calibration_intercept','calibration_slope'] if calibration else [])}})
    draws=pd.DataFrame(samples);rows=[]
    for name in prob_cols:
        point=metrics(y,frame[name],cal=calibration);sel=draws[draws.model.eq(name)]
        for metric in sel.columns.difference(['replicate','model']):
            lo,hi=sel[metric].quantile([.025,.975]);rows.append({'model':name,'metric':metric,'estimate':point[metric],'lower':lo,'upper':hi,'resamples':n_boot,'unit':'person','unique_clusters':len(ids)})
    return pd.DataFrame(rows),draws
