"""Additional diagnostic analyses; post hoc sensitivities retain exploratory status."""
from __future__ import annotations
import argparse,hashlib,importlib.util,itertools
import joblib
from common import *

FOLDS=[(7,9,11),(10,12,14),(13,15,17),(16,18,20)]
CALENDAR=['survey_year_centered','post_2018_workweek_reform','covid_period','post_covid_period']
_spec=importlib.util.spec_from_file_location('inference',ROOT/'python/cluster_inference.py');_inf=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_inf)

def group(pid):return int(hashlib.sha256(('qq20261026:'+str(int(pid))).encode()).hexdigest()[:16],16)%100
def fit_cached(key,name,train,cols=None):
    md=PRIV/'models';md.mkdir(exist_ok=True);p=md/(key+'.joblib')
    if p.exists():return joblib.load(p)
    f=fit_model(name,train,cols,{'threads':3});joblib.dump(f,p);return f
def threshold(name,method):
    a=pd.read_csv(RES/'selected_thresholds.csv');return float(a[(a.model==name)&(a.method==method)].threshold.iloc[0])
def paired(frame,cols,clusters,nboot=500,seed=SEED+83):
    labels,code=np.unique(np.asarray(clusters),return_inverse=True);scores={c:_inf.FastScore(frame.exit_label_t1,frame[c]) for c in cols};rng=np.random.default_rng(seed);rows=[]
    for rep in range(nboot):
        w=rng.multinomial(len(labels),np.full(len(labels),1/len(labels)))[code].astype(float)
        for c,s in scores.items():rows.append({'replicate':rep,'model':c,**s.score(w,False)})
    dr=pd.DataFrame(rows);ci=[];dif=[]
    for c in cols:
        point=scores[c].score(np.ones(len(frame)),False);z=dr[dr.model.eq(c)]
        for m in ['roc_auc','pr_auc','brier']:
            lo,hi=z[m].quantile([.025,.975]);ci.append({'model':c,'metric':m,'estimate':point[m],'lower':lo,'upper':hi,'resamples':nboot,'clusters':len(labels)})
    for a,b in itertools.combinations(cols,2):
        aa=dr[dr.model.eq(a)].set_index('replicate');bb=dr[dr.model.eq(b)].set_index('replicate')
        for m in ['roc_auc','pr_auc','brier']:
            lo,hi=(aa[m]-bb[m]).quantile([.025,.975]);delta=scores[a].score(np.ones(len(frame)),False)[m]-scores[b].score(np.ones(len(frame)),False)[m]
            dif.append({'model_A':a,'model_B':b,'metric':m,'difference_A_minus_B':delta,'lower':lo,'upper':hi,'resamples':nboot,'clusters':len(labels)})
    return pd.DataFrame(ci),pd.DataFrame(dif),dr

def audit():
    d=load_data();tr=d[d.wave.le(20)];cols=feature_names(tr);rows=[]
    for c in cols:
        for part,g in d.groupby('split'):
            x=g[c];rows.append({'feature':c,'partition':part,'observed':int(x.notna().sum()),'distinct_nonmissing':x.nunique(dropna=True),'constant_in_primary_training':tr[c].nunique(dropna=True)<=1,'minimum':pd.to_numeric(x,errors='coerce').min(),'maximum':pd.to_numeric(x,errors='coerce').max()})
    save(pd.DataFrame(rows),'feature_variation.csv')
    out=[]
    for end,start,stop in FOLDS:
        fc=feature_names(d[d.wave.le(end)])
        for c in sorted(set(cols)-set(fc)):out.append({'train_end':end,'feature_excluded_beyond_primary':c,'observed_in_fold':int(d.loc[d.wave.le(end),c].notna().sum())})
    save(pd.DataFrame(out),'fold_extra_exclusions.csv')
    print('EXTENDED AUDIT COMPLETE',flush=True)

def drift():
    d=load_data().copy();d['age_band']=pd.cut(d.age_final,[0,35,50,65,np.inf],right=False,labels=['below35','35to49','50to64','65plus']).astype(str)
    assert d.age_final.notna().all()
    d['stratum']=d.age_band+'|'+d.cohort;parts=['train','validation','test'];strata=sorted(d.stratum.unique());K=len(strata)
    code=d.stratum.map({s:i for i,s in enumerate(strata)}).to_numpy();pc=d['split'].map({s:i for i,s in enumerate(parts)}).to_numpy();y=d.exit_label_t1.to_numpy(float)
    ids,ic=np.unique(d.pid,return_inverse=True);ix=pc*K+code
    def tables(w):
        n=np.bincount(ix,weights=w,minlength=3*K).reshape(3,K);e=np.bincount(ix,weights=w*y,minlength=3*K).reshape(3,K)
        r=np.divide(e,n,out=np.full_like(n,np.nan),where=n>0);return n,e,r
    n,e,r=tables(np.ones(len(d)));common=np.all(n>0,axis=0);q=n[:,common].sum(axis=0);q/=q.sum();commoncols=np.flatnonzero(common)
    def estimate(nn,ee,rr):
        raw=ee.sum(axis=1)/nn.sum(axis=1);nc=nn[:,common];rc=rr[:,common];p=nc/nc.sum(axis=1,keepdims=True);cr=np.sum(p*rc,axis=1);std=rc@q;out=[]
        for a,b in [(0,1),(1,2),(0,2)]:
            C=np.sum((p[b]-p[a])*(rc[a]+rc[b])/2);W=np.sum((rc[b]-rc[a])*(p[a]+p[b])/2);U=(raw[b]-cr[b])-(raw[a]-cr[a])
            assert np.isclose(raw[b]-raw[a],C+W+U,atol=1e-12)
            out.append([raw[b]-raw[a],C,W,U])
        return raw,cr,std,np.array(out)
    raw,cr,std,decomp=estimate(n,e,r);rng=np.random.default_rng(SEED+84);std_draws=[];delta_draws=[]
    for _ in range(500):
        w=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))[ic];nn,ee,rr=tables(w)
        if (nn[:,common]<=0).any():raise RuntimeError('Bootstrap common-support cell empty')
        _,_,ss,dd=estimate(nn,ee,rr);std_draws.append(ss);delta_draws.append(dd)
    std_draws=np.asarray(std_draws);delta_draws=np.asarray(delta_draws);summary=[]
    for j,part in enumerate(parts):
        g=d[d['split'].eq(part)];lo,hi=np.quantile(std_draws[:,j],[.025,.975]);summary.append({'partition':part,'n':len(g),'events':int(g.exit_label_t1.sum()),'raw_event_rate':raw[j],'common_support_n':int(n[j,common].sum()),'common_support_coverage':n[j,common].sum()/n[j].sum(),'common_support_event_rate':cr[j],'standardized_event_rate':std[j],'standardized_lower':lo,'standardized_upper':hi,'mean_age':g.age_final.mean(),'share_2018_cohort':g.cohort.str.startswith('18').mean(),'nonemployment_share_of_events':g.destination.eq('nonemployed').sum()/g.exit_label_t1.sum()})
    save(pd.DataFrame(summary),'drift_standardization.csv');out=[]
    components=['total_event_rate_change','age_cohort_composition_common_support','within_stratum_rate_change_common_support','outside_common_support_contribution']
    for j,(a,b) in enumerate([(0,1),(1,2),(0,2)]):
        for k,m in enumerate(components):
            lo,hi=np.quantile(delta_draws[:,j,k],[.025,.975]);out.append({'from_partition':parts[a],'to_partition':parts[b],'component':m,'estimate':decomp[j,k],'lower':lo,'upper':hi,'resamples':500})
    save(pd.DataFrame(out),'drift_decomposition.csv');detail=[]
    for j,part in enumerate(parts):
        for k,s in enumerate(strata):detail.append({'partition':part,'stratum':s,'n':n[j,k],'events':e[j,k],'event_rate':r[j,k],'common_support':bool(common[k]),'reference_weight':q[list(commoncols).index(k)] if common[k] else np.nan})
    save(pd.DataFrame(detail),'drift_strata.csv')
    dump({'age_cutpoints':[0,35,50,65,'infinity'],'common_support':'Age-by-cohort cells observed in all three partitions','reference':'Pooled person-wave distribution of common-support cells, held fixed during bootstrap','decomposition':'Symmetric Kitagawa decomposition within common support, plus exact full-minus-common residual; no historical 2018-cohort rate imputed','resampling':'500 paired person-cluster draws spanning all partitions','interpretation':'Descriptive decomposition; neither causal effects nor a correction of follow-up selection'},RES/'drift_design.json')
    print('EXTENDED DRIFT COMPLETE',flush=True)

def origins():
    d=load_data();oof=pd.read_pickle(PRIV/'forward_oof.pkl');rows=[];cis=[];contrasts=[];foldrows=[]
    for end in [14,17]:
        tr=d[d.wave.le(end)];va=d[d.wave.between(end+1,end+3)];te=d[d.wave.between(end+4,end+6)];v=va[['pid','exit_label_t1']].copy();t=te[['pid','exit_label_t1']].copy()
        for name in BASES:
            f=fit_cached(f'rolling_{end}_{name}',name,tr);v[name]=predict(f,va);t[name]=predict(f,te);del f
        oo=oof[oof.wave.le(end)];assert oo.wave.max()+1<=va.wave.min()
        meta=LogisticRegression(C=1,max_iter=2000).fit(oo[BASES],oo.exit_label_t1.astype(int));v['forward_stack']=meta.predict_proba(v[BASES])[:,1];t['forward_stack']=meta.predict_proba(t[BASES])[:,1]
        for fe,fs,ff in FOLDS:
            if ff<=end:foldrows.append({'origin_train_end':end,'base_train_end':fe,'latest_base_outcome':fe+1,'oof_start':fs,'oof_end':ff,'meta_training_n':len(oo)})
        for name in BASES+['forward_stack']:
            cal=calibrators(v[name],va.exit_label_t1)
            for method in ['raw','platt','isotonic']:
                pv=calibrated(cal,v[name],method);pt=calibrated(cal,t[name],method);rows.append({'train_end':end,'model':name,'method':method,'train_n':len(tr),'valid_n':len(va),'meta_n':len(oo) if name=='forward_stack' else np.nan,**metrics(te.exit_label_t1,pt,f1_threshold(va.exit_label_t1,pv))})
        ci,co,dr=paired(t,BASES+['forward_stack'],t.pid,500,SEED+end);cis.append(ci.assign(train_end=end));contrasts.append(co.assign(train_end=end));save(dr.assign(train_end=end),f'origin{end}_bootstrap_replicates.csv');t.to_pickle(PRIV/f'origin{end}_test.pkl')
        save(pd.DataFrame(rows),'all_model_origins.csv');print('EXTENDED ORIGIN COMPLETE',end,flush=True)
    save(pd.concat(cis),'origin_intervals.csv');save(pd.concat(contrasts),'origin_contrasts.csv');save(pd.DataFrame(foldrows),'origin_forward_folds.csv')

def prediction_audits():
    d=load_data();tr=d[d.wave.le(20)];va=d[d.wave.between(21,23)];te=d[d.wave.between(24,26)];v=pd.read_pickle(PRIV/'valid_predictions.pkl');t=pd.read_pickle(PRIV/'test_predictions.pkl');rows=[]
    for name in ['catboost','forward_stack']:
        for method in ['raw','platt']:
            col=name if method=='raw' else name+'_'+method;g=t[t.age_final.lt(65)]
            rows.append({'fit_population':'all ages','evaluation_population':'below 65','model':name,'method':method,**metrics(g.exit_label_t1,g[col],threshold(name,method))})
    f=fit_cached('under65_refit','catboost',tr[tr.age_final.lt(65)]);vv=va[va.age_final.lt(65)];tt=te[te.age_final.lt(65)];pv=predict(f,vv);pt=predict(f,tt);cal=calibrators(pv,vv.exit_label_t1)
    for method in ['raw','platt']:rows.append({'fit_population':'below 65','evaluation_population':'below 65','model':'catboost','method':method,**metrics(tt.exit_label_t1,calibrated(cal,pt,method),f1_threshold(vv.exit_label_t1,calibrated(cal,pv,method)))})
    save(pd.DataFrame(rows),'matched_below65.csv');assert len(tt)==int(rows[0]['n'])
    paired_age=t.loc[tt.index,['pid','exit_label_t1','catboost','forward_stack']].copy();paired_age['under65_refit']=pt
    age_ci,age_contrast,age_draws=paired(paired_age,['catboost','under65_refit','forward_stack'],paired_age.pid,500,SEED+65)
    save(age_ci,'matched_below65_intervals.csv');save(age_contrast,'matched_below65_contrasts.csv');save(age_draws,'matched_below65_bootstrap_replicates.csv')
    # Honest same-model comparison under a randomized hash allocation. The 70%
    # arm also contains people first observed later; seen status is separated.
    gp=d.pid.map(group);a=d[d.wave.le(20)&gp.lt(70)];b=d[d.wave.between(21,23)&gp.between(70,84)];f=fit_cached('person_disjoint_refit','catboost',a);pc=predict(f,b);cal=calibrators(pc,b.exit_label_t1);tg=te.pid.map(group);ps=predict(f,te);seen=te.pid.isin(set(a.pid));out=[]
    for label,mask in [('hash_train_70_all_test_rows',tg.lt(70)),('hash_test_15_all_test_rows',tg.ge(85)),('hash_train_70_actually_seen',tg.lt(70)&seen),('hash_train_70_new',tg.lt(70)&~seen)]:
        g=te[mask]
        for method in ['raw','platt']:out.append({'evaluation_group':label,'model':'disjoint_catboost','method':method,'people':g.pid.nunique(),**metrics(g.exit_label_t1,calibrated(cal,ps[mask],method),f1_threshold(b.exit_label_t1,calibrated(cal,pc,method)))})
    save(pd.DataFrame(out),'disjoint_same_model_groups.csv')
    q=te[tg.ge(85)&te.exit_label_t1.eq(1)];z=te[te.destination.isin(['self_employed','unpaid_family'])];dump({'disjoint_events':len(q),'nonwage_events':len(z),'intersecting_person_wave_keys':len(set(zip(q.pid,q.wave))&set(zip(z.pid,z.wave))),'identical_event_sets':set(zip(q.pid,q.wave))==set(zip(z.pid,z.wave))},RES/'389_event_audit.json')
    # Store raw-scale equivalents and exact selected-set identities.
    out=[];predsets={}
    for source,vg in [('all_21_23',v),('pre2020_21_22',v[v.wave.le(22)]),('2020_only_23',v[v.wave.eq(23)])]:
        for name in ['catboost','logistic','forward_stack']:
            cc=calibrators(vg[name],vg.exit_label_t1);pp=calibrated(cc,vg[name],'platt');cut=f1_threshold(vg.exit_label_t1,pp);coef=float(cc['platt'].coef_[0,0]);inter=float(cc['platt'].intercept_[0]);rawcut=float(expit((logit(cut)-inter)/coef));selected=calibrated(cc,t[name],'platt')>=cut;predsets[(source,name)]=selected
            out.append({'calibration_window':source,'model':name,'platt_threshold':cut,'raw_equivalent_threshold':rawcut,'platt_intercept':inter,'platt_slope':coef,'test_selected':int(selected.sum()),'test_selected_matches_all_window':None})
    for r in out:r['test_selected_matches_all_window']=bool(np.array_equal(predsets[(r['calibration_window'],r['model'])],predsets[('all_21_23',r['model'])]))
    save(pd.DataFrame(out),'calibration_raw_thresholds.csv')
    print('EXTENDED PREDICTION AUDITS COMPLETE',flush=True)

def logit_stack():
    oo=pd.read_pickle(PRIV/'forward_oof.pkl');v=pd.read_pickle(PRIV/'valid_predictions.pkl');t=pd.read_pickle(PRIV/'test_predictions.pkl');z=lambda f:logit(np.clip(f[BASES].to_numpy(),1e-6,1-1e-6))
    meta=LogisticRegression(C=1,max_iter=2000).fit(z(oo),oo.exit_label_t1.astype(int));pv=meta.predict_proba(z(v))[:,1];pt=meta.predict_proba(z(t))[:,1];cal=calibrators(pv,v.exit_label_t1);rows=[]
    for method in ['raw','platt','isotonic']:rows.append({'model':'logit_input_stack','method':method,**metrics(t.exit_label_t1,calibrated(cal,pt,method),f1_threshold(v.exit_label_t1,calibrated(cal,pv,method)))})
    save(pd.DataFrame(rows),'logit_stack.csv');save(pd.DataFrame({'base_model':BASES,'coefficient':meta.coef_[0],'intercept':meta.intercept_[0]}),'logit_stack_coefficients.csv')
    original=joblib.load(PRIV/'models/forward_stack.joblib');assert np.all(original.coef_[0]>0);floor=expit(original.intercept_[0]);out=[]
    for name,prob in [('probability_input_raw',t.forward_stack.to_numpy()),('probability_input_platt',t.forward_stack_platt.to_numpy()),('probability_input_isotonic',t.forward_stack_isotonic.to_numpy()),('logit_input_raw',pt),('logit_input_platt',calibrated(cal,pt,'platt')),('logit_input_isotonic',calibrated(cal,pt,'isotonic'))]:
        cut=np.quantile(prob,.1);mask=prob<=cut;out.append({'variant':name,'minimum_probability':prob.min(),'lower_decile_cutoff':cut,'lower_decile_n':int(mask.sum()),'lower_decile_mean_prediction':prob[mask].mean(),'lower_decile_observed_rate':t.exit_label_t1.to_numpy()[mask].mean()})
    save(pd.DataFrame(out),'stack_lower_tail.csv');dump({'probability_input_intercept':float(original.intercept_[0]),'all_coefficients_positive':True,'mathematical_minimum_over_unit_cube':float(floor),'observed_minimum':float(t.forward_stack.min()),'logit_input_clip':1e-6,'meta_C':1,'meta_standardization':False,'interpretation':'Exploratory scale sensitivity; no primary model replacement based on test scores'},RES/'stack_floor.json')
    t['logit_input_stack']=pt;ci,co,dr=paired(t,['catboost','forward_stack','logit_input_stack'],t.pid,500);save(ci,'logit_stack_intervals.csv');save(co,'logit_stack_contrasts.csv');save(dr,'logit_stack_bootstrap_replicates.csv')
    print('EXTENDED LOGIT STACK COMPLETE',flush=True)

def precision_audit():
    """Show why post-hoc map precision must be consistent across models/runs."""
    v=pd.read_pickle(PRIV/'base_valid_predictions.pkl');t=pd.read_pickle(PRIV/'base_test_predictions.pkl');out=[]
    for dtype in ['float32','float64']:
        iso=IsotonicRegression(out_of_bounds='clip').fit(v.xgboost.to_numpy().astype(dtype),v.exit_label_t1.to_numpy(int))
        pp=iso.predict(t.xgboost.to_numpy().astype(dtype));mm=metrics(t.exit_label_t1,pp)
        out.append({'fit_and_input_dtype':dtype,'distinct_test_scores':np.unique(pp).size,**{k:mm[k] for k in ['roc_auc','pr_auc','brier','recall20']}})
    save(pd.DataFrame(out),'calibration_precision_audit.csv')
    print('EXTENDED CALIBRATION PRECISION AUDIT COMPLETE',flush=True)

def household():
    t=pd.read_pickle(PRIV/'test_predictions.pkl');pids=t.pid.unique();parent={p:p for p in pids}
    def root(p):
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    valid=t[t.hhid.notna()&t.hhid.gt(0)]
    for _,g in valid.groupby(['wave','hhid']):
        ids=g.pid.to_numpy()
        for p in ids[1:]:union(ids[0],p)
    groups=t.pid.map(lambda p:root(p));cols=['catboost','forward_stack','catboost_platt','forward_stack_platt'];ci,co,dr=paired(t,cols,groups,1000,SEED+91);save(ci,'household_component_intervals.csv');save(co,'household_component_contrasts.csv');save(dr,'household_bootstrap_replicates.csv')
    size=t.assign(component=groups).groupby('component').agg(rows=('pid','size'),people=('pid','nunique'));assert t.assign(component=groups).groupby('pid').component.nunique().max()==1
    dump({'people':len(pids),'connected_components':len(size),'maximum_people_per_component':int(size.people.max()),'maximum_rows_per_component':int(size.rows.max()),'median_people':float(size.people.median()),'singleton_person_components':int(size.people.eq(1).sum()),'rows_missing_household':int(len(t)-len(valid)),'resamples':1000,'construction':'Union of people co-resident in a positive household identifier within any same test wave; all rows of each person retained in one component','scope':'Dependence sensitivity, not full sampling-design or population inference'},RES/'household_design.json')
    print('EXTENDED HOUSEHOLD COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','drift','origins','predictions','logit','precision','household']);args=ap.parse_args();{'audit':audit,'drift':drift,'origins':origins,'predictions':prediction_audits,'logit':logit_stack,'precision':precision_audit,'household':household}[args.stage]()
