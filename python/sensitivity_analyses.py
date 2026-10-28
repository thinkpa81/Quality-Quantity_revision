"""Aggregate audits and prespecified-after-review diagnostic reanalyses.
All decisions and searches use development data; sensitivities are exploratory.
"""
from __future__ import annotations
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
import joblib
from common import *
from feature_definitions import FEATURE_GROUPS

CALENDAR=['survey_year_centered','post_2018_workweek_reform','covid_period','post_covid_period']
RELATIVE=['wage_percentile_wave','hours_percentile_wave','wage_to_wave_median','wage_to_industry_wave_median','wage_to_occupation_wave_median']

def audits():
    d=load_data();risk=pd.read_pickle(PRIV/'risk.pkl');panel=pd.read_pickle(PRIV/'panel.pkl')
    rows=[]
    for part,g in risk[risk.wave.le(26)].groupby('split'):
        a=g[g.outcome_observed];b=g[~g.outcome_observed]
        for col,kind in [('age_final','mean'),('monthly_wage','mean'),('tenure_years','mean'),('health_status','mean'),('household_size_raw','mean'),('gender','female'),('region_changed_lag1','mean')]:
            x=pd.to_numeric(a[col],errors='coerce');z=pd.to_numeric(b[col],errors='coerce')
            if kind=='female':x=a[col].eq('female').astype(float).where(a[col].notna());z=b[col].eq('female').astype(float).where(b[col].notna())
            denom=np.sqrt((x.var()+z.var())/2);rows.append({'partition':part,'characteristic':col+'_'+kind,'retained_n':len(a),'excluded_n':len(b),'retained_observed':int(x.notna().sum()),'excluded_observed':int(z.notna().sum()),'retained_mean':x.mean(),'excluded_mean':z.mean(),'standardized_difference_excluded_minus_retained':(z.mean()-x.mean())/denom if denom>0 else np.nan})
        n=len(g);events=a.exit_label_t1.sum();rows.append({'partition':part,'characteristic':'event_prevalence_bounds','retained_n':len(a),'excluded_n':len(b),'retained_mean':events/n,'excluded_mean':(events+len(b))/n})
    save(pd.DataFrame(rows),'followup_selection_baseline.csv')
    routing=[c for c in panel if c.startswith('routing_')]
    out=[]
    for col in routing:
        k=panel[panel.state.eq('nonemployed')].groupby(['wave',col],dropna=False).size().reset_index(name='n').rename(columns={col:'value'});k['variable']=col;out.append(k)
    save(pd.concat(out),'nonemployment_routing_audit.csv')
    # Nonemployment is defined by questionnaire routing, not by retirement/unemployment detail.
    # Actual last-week-work answers provide a separate contradiction audit, without changing the estimand.
    audit=panel.groupby(['wave','state'],observed=True).agg(n=('pid','size'),fallback_n=('state_fallback','sum')).reset_index()
    save(audit,'state_classification_by_wave.csv')
    tr=d[d.wave.le(20)];va=d[d.wave.between(21,23)];te=d[d.wave.between(24,26)]
    overlap=[]
    for part,g in [('validation',va),('test',te)]:
        seen=set(tr.pid) if part=='validation' else set(tr.pid)|set(va.pid)
        overlap.append({'partition':part,'rows':len(g),'people':g.pid.nunique(),'seen_rows':g.pid.isin(seen).sum(),'seen_people':g[g.pid.isin(seen)].pid.nunique(),'new_rows':(~g.pid.isin(seen)).sum(),'new_people':g[~g.pid.isin(seen)].pid.nunique()})
    save(pd.DataFrame(overlap),'person_overlap.csv')
    print('AUDITS COMPLETE',flush=True)

def sensitivities():
    d=load_data();tr=d[d.wave.le(20)];va=d[d.wave.between(21,23)];te=d[d.wave.between(24,26)]
    md=PRIV/'models';md.mkdir(exist_ok=True);rows=[];details=[]
    def fit(key,train,drop=(),name='catboost',config=None):
        path=md/(key+'.joblib')
        if path.exists():return joblib.load(path)
        f=fit_model(name,train,feature_names(train,drop),config);joblib.dump(f,path);return f
    def evaluate(key,train,valid,test,drop=(),name='catboost',config=None,subset_note='same outcome'):
        f=fit(key,train,drop,name,config);vp=predict(f,valid);tp=predict(f,test);cal=calibrators(vp,valid.exit_label_t1)
        for method in ['raw','platt']:
            pv=calibrated(cal,vp,method);pt=calibrated(cal,tp,method);th=f1_threshold(valid.exit_label_t1,pv)
            rows.append({'analysis':key,'model':name,'method':method,'train_n':len(train),'valid_n':len(valid),'features':len(f['cols']),'scope':subset_note,**metrics(test.exit_label_t1,pt,th)})
        save(pd.DataFrame(rows),'sensitivity_models.csv');print('SENSITIVITY',key,flush=True)
        return f,vp,tp
    for key,drop in [('no_year',['survey_year_centered']),('no_calendar',CALENDAR),('no_wave_relative',RELATIVE),('no_history',FEATURE_GROUPS['panel_history'] if 'panel_history' in FEATURE_GROUPS else [c for c in feature_names(tr) if 'lag1' in c or c.startswith('prior_') or 'rolling3' in c or c in ['waves_since_first_observed','panel_observation_number']])]:
        evaluate(key,tr,va,te,drop)
    evaluate('under65_refit',tr[tr.age_final.lt(65)],va[va.age_final.lt(65)],te[te.age_final.lt(65)],subset_note='Restricted to age below 65 at each baseline')
    # A genuinely disjoint split of people, with temporal windows retained.
    def group(pid):return int(hashlib.sha256(('qq20261026:'+str(int(pid))).encode()).hexdigest()[:16],16)%100
    assign={p:group(p) for p in d.pid.unique()};grp=d.pid.map(assign)
    a=d[d.wave.le(20)&grp.lt(70)];b=d[d.wave.between(21,23)&grp.between(70,84)];c=d[d.wave.between(24,26)&grp.ge(85)]
    assert not(set(a.pid)&set(b.pid) or set(a.pid)&set(c.pid) or set(b.pid)&set(c.pid))
    evaluate('person_disjoint_refit',a,b,c,subset_note='SHA256 person partition 70/15/15 and chronological windows; no person crosses partitions')
    save(pd.DataFrame([{'partition':n,'rows':len(g),'people':g.pid.nunique(),'events':int(g.exit_label_t1.sum())} for n,g in [('train',a),('validation',b),('test',c)]]),'person_disjoint_sample.csv')
    # Excluding questionnaire-routed nonemployment changes the target to non-wage work among employed.
    evaluate('nonwage_employed_only',tr[tr.destination.ne('nonemployed')],va[va.destination.ne('nonemployed')],te[te.destination.ne('nonemployed')],subset_note='Non-wage employment versus continued wage employment, excluding nonemployment')
    evaluate('nonemployment_only',tr[tr.destination.isin(['wage','nonemployed'])],va[va.destination.isin(['wage','nonemployed'])],te[te.destination.isin(['wage','nonemployed'])],subset_note='Nonemployment versus continued wage employment, excluding non-wage employment')
    # Directly fulfill the missing-detail exclusion diagnostic; the preceding sensitivity is its substantive counterpart.
    evaluate('exclude_missing_status',tr[tr.next_employee_status_raw.notna()],va[va.next_employee_status_raw.notna()],te[te.next_employee_status_raw.notna()],subset_note='Excludes next-wave missing worker-status detail; structurally removes almost all nonemployment')
    evaluate('strict_boundary_embargo',d[d.wave.le(19)],d[d.wave.between(21,22)],te,subset_note='Latest training outcome20 < validation21; latest calibration outcome23 < test24')
    for end in [14,17]:
        for name in ['catboost','logistic']:
            evaluate(f'rolling_{end}_{name}',d[d.wave.le(end)],d[d.wave.between(end+1,end+3)],d[d.wave.between(end+4,end+6)],name=name,subset_note=f'Expanding origin: training1-{end}; calibration{end+1}-{end+3}; test{end+4}-{end+6}')
    evaluate('balanced_logistic',tr,va,te,name='logistic',config={'class_weight':'balanced'})
    # Modest finite searches, chosen only by validation AP (Brier as deterministic tie-breaker).
    tuning=[]
    for name,configs in [('catboost',[{'depth':4,'iterations':300},{'depth':6,'iterations':300},{'depth':8,'iterations':300},{'depth':6,'iterations':500}]),('logistic',[{'C':.1},{'C':1},{'C':10,'max_iter':10000}])]:
        candidates=[]
        for k,cfg in enumerate(configs):
            key=f'tuned_{name}_{k}'
            if cfg in [{'depth':6,'iterations':300},{'C':1}]:f=joblib.load(md/(name+'.joblib'))
            else:f=fit(key,tr,name=name,config=cfg)
            vp=predict(f,va);m=metrics(va.exit_label_t1,vp,cal=False);candidates.append((m['pr_auc'],-m['brier'],k,f,vp));tuning.append({'model':name,'candidate':k,'config':json.dumps(cfg),'validation_ap':m['pr_auc'],'validation_brier':m['brier']})
        best=max(candidates,key=lambda x:x[:2]);_,_,k,f,vp=best;tp=predict(f,te);cal=calibrators(vp,va.exit_label_t1)
        for method in ['raw','platt']:
            pv=calibrated(cal,vp,method);pt=calibrated(cal,tp,method);rows.append({'analysis':'validation_tuned_'+name,'model':name,'method':method,'train_n':len(tr),'valid_n':len(va),'features':len(f['cols']),'scope':f'Validation AP selected candidate {k}',**metrics(te.exit_label_t1,pt,f1_threshold(va.exit_label_t1,pv))})
        for r in tuning:
            if r['model']==name:r['selected']=r['candidate']==k
        save(pd.DataFrame(rows),'sensitivity_models.csv');save(pd.DataFrame(tuning),'tuning_validation.csv')
    # Meta-learner regularization search; base fits and OOF records remain identical.
    oof=pd.read_pickle(PRIV/'forward_oof.pkl');v=pd.read_pickle(PRIV/'valid_predictions.pkl');t=pd.read_pickle(PRIV/'test_predictions.pkl');cand=[]
    for cval in [.1,1,10]:
        meta=LogisticRegression(C=cval,max_iter=2000).fit(oof[BASES],oof.exit_label_t1.astype(int));pv=meta.predict_proba(v[BASES])[:,1];m=metrics(v.exit_label_t1,pv,cal=False)
        cand.append((m['pr_auc'],-m['brier'],cval,meta,pv));tuning.append({'model':'forward_stack','candidate':cval,'config':json.dumps({'C':cval}),'validation_ap':m['pr_auc'],'validation_brier':m['brier']})
    _,_,cval,meta,pv=max(cand,key=lambda x:x[:2]);pt=meta.predict_proba(t[BASES])[:,1];cal=calibrators(pv,v.exit_label_t1)
    for method in ['raw','platt']:
        rows.append({'analysis':'validation_tuned_forward_stack','model':'forward_stack','method':method,'train_n':len(tr),'valid_n':len(va),'features':4,'scope':f'Validation selected C={cval}',**metrics(t.exit_label_t1,calibrated(cal,pt,method),f1_threshold(v.exit_label_t1,calibrated(cal,pv,method)))})
    for r in tuning:
        if r['model']=='forward_stack':r['selected']=r['candidate']==cval
    save(pd.DataFrame(rows),'sensitivity_models.csv');save(pd.DataFrame(tuning),'tuning_validation.csv')
    print('MODEL SENSITIVITIES COMPLETE',flush=True)

def predictions():
    d=load_data();v=pd.read_pickle(PRIV/'valid_predictions.pkl');t=pd.read_pickle(PRIV/'test_predictions.pkl');tr=d[d.wave.le(20)]
    thresholds=pd.read_csv(RES/'selected_thresholds.csv');th=lambda n,m:float(thresholds[(thresholds.model==n)&(thresholds.method==m)].threshold.iloc[0])
    names=BASES+['forward_stack'];rows=[]
    for wave,g in t.groupby('wave'):
        for name in names:
            for meth in ['raw','platt','isotonic']:
                col=name if meth=='raw' else name+'_'+meth;rows.append({'wave':wave,'year':1997+wave,'model':name,'method':meth,**metrics(g.exit_label_t1,g[col],th(name,meth))})
    save(pd.DataFrame(rows),'test_wave_metrics.csv')
    rows=[]
    for source,vg in [('all_21_23',v),('pre2020_21_22',v[v.wave.le(22)]),('2020_only_23',v[v.wave.eq(23)])]:
        for name in ['catboost','logistic','forward_stack']:
            cal=calibrators(vg[name],vg.exit_label_t1)
            pt=calibrated(cal,t[name],'platt');pv=calibrated(cal,vg[name],'platt')
            rows.append({'calibration_window':source,'calibration_n':len(vg),'calibration_prevalence':vg.exit_label_t1.mean(),'model':name,**metrics(t.exit_label_t1,pt,f1_threshold(vg.exit_label_t1,pv))})
    save(pd.DataFrame(rows),'calibration_window_sensitivity.csv')
    # Subgroups share the same overall validation-selected operating point.
    tt=t.copy();tt['age_group']=pd.cut(tt.age_final,[0,35,50,65,np.inf],right=False,labels=['below35','35to49','50to64','65plus']).astype(str)
    rows=[]
    for variable in ['age_group','gender','region','cohort']:
        for group,g in tt.groupby(variable,dropna=False):
            for name in ['catboost','logistic','forward_stack']:
                for meth in ['raw','platt']:
                    col=name if meth=='raw' else name+'_'+meth
                    rows.append({'subgroup_variable':variable,'group':group,'model':name,'method':meth,'event_share':g.exit_label_t1.sum()/tt.exit_label_t1.sum(),**metrics(g.exit_label_t1,g[col],th(name,meth))})
    save(pd.DataFrame(rows),'subgroup_metrics.csv')
    seen=set(d[d.wave.le(23)].pid);rows=[]
    for label,g in [('new',t[~t.pid.isin(seen)]),('seen',t[t.pid.isin(seen)])]:
        for name in names:
            rows.append({'population':label,'model':name,'method':'raw',**metrics(g.exit_label_t1,g[name],th(name,'raw'))})
    save(pd.DataFrame(rows),'new_person_subset_metrics.csv')
    rows=[]
    for wave,g in [('pooled',t),*list(t.groupby('wave'))]:
        g=g[g.weight18_cross.notna() & g.weight18_cross.gt(0)]
        for name in names:
            rows.append({'wave':wave,'model':name,'weight':'2018_integrated_cross_sectional','effective_n_kish':g.weight18_cross.sum()**2/(g.weight18_cross**2).sum(),**metrics(g.exit_label_t1,g[name],th(name,'raw'),weights=g.weight18_cross)})
    save(pd.DataFrame(rows),'survey_weighted_metrics.csv')
    # Outcome-specific scores from the primary fit as a descriptive counterpart to target-specific refits.
    rows=[]
    for dest in ['nonemployed','self_employed','unpaid_family']:
        g=t[t.destination.isin(['wage',dest])]
        rows.append({'destination':dest,'model':'catboost',**metrics(g.exit_label_t1,g.catboost,th('catboost','raw'))})
    save(pd.DataFrame(rows),'destination_specific_scores.csv')
    # Identical source table for manuscript and supplementary utility reports.
    rows=[]
    for name in ['catboost','logistic','forward_stack']:
        for meth in ['raw','platt']:
            col=name if meth=='raw' else name+'_'+meth
            for label,benefit,fp_cost,fn_cost in [('low',10,1,5),('medium',20,5,10),('high',50,20,15)]:
                pv=v[col].to_numpy();y=v.exit_label_t1.to_numpy(int);order=np.argsort(-pv,kind='stable');ys=y[order];ps=pv[order];tp=np.cumsum(ys);fp=np.arange(1,len(y)+1)-tp;fn=y.sum()-tp
                u=benefit*tp-fp_cost*fp-fn_cost*fn
                # Only include complete tied-score groups, plus the treat-none threshold.
                ends=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(ps)-1];options=np.r_[-fn_cost*y.sum(),u[ends]];which=int(np.argmax(options));cut=1+1e-9 if which==0 else ps[ends[which-1]]
                yt=t.exit_label_t1.to_numpy(int);pred=t[col].to_numpy()>=cut;TP=int((yt*pred).sum());FP=int(((1-yt)*pred).sum());FN=int(yt.sum()-TP)
                rows.append({'model':name,'method':meth,'scenario':label,'benefit_TP':benefit,'cost_FP':fp_cost,'cost_FN':fn_cost,'threshold':cut,'test_TP':TP,'test_FP':FP,'test_FN':FN,'utility_units':benefit*TP-fp_cost*FP-fn_cost*FN,'utility_per_1000':1000*(benefit*TP-fp_cost*FP-fn_cost*FN)/len(t),'treat_none_utility':-fn_cost*int(yt.sum()),'treat_all_utility':benefit*int(yt.sum())-fp_cost*int((1-yt).sum())})
    save(pd.DataFrame(rows),'utility_scenarios.csv')
    rows=[];y=t.exit_label_t1.to_numpy(int);N=len(y)
    for cut in np.arange(.01,.501,.01):
        for name,col in [('CatBoost raw','catboost'),('CatBoost Platt','catboost_platt'),('Logistic Platt','logistic_platt'),('Forward stack Platt','forward_stack_platt')]:
            pred=t[col].to_numpy()>=cut;tp=(pred*y).sum();fp=(pred*(1-y)).sum();rows.append({'threshold':cut,'model':name,'net_benefit':tp/N-fp/N*cut/(1-cut)})
        rows.extend([{'threshold':cut,'model':'Treat all','net_benefit':y.mean()-(1-y.mean())*cut/(1-cut)},{'threshold':cut,'model':'Treat none','net_benefit':0}])
    save(pd.DataFrame(rows),'decision_curve.csv')
    print('PREDICTION DIAGNOSTICS COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audits','models','predictions']);a=ap.parse_args()
    {'audits':audits,'models':sensitivities,'predictions':predictions}[a.stage]()
