"""Post hoc history-stratified evaluation of one fixed person-disjoint model.

No model or threshold is selected with these test outcomes. Hash arms are
stratified by *potential* supervised training eligibility, defined identically
before consulting assignment. A secondary definition uses any panel interview.
All contrasts condition on the fitted model and observed follow-up.
"""
from __future__ import annotations
import hashlib,json
import joblib
from common import *
from cluster_inference import FastScore

def group(pid):
    return int(hashlib.sha256(('qq20261026:'+str(int(pid))).encode()).hexdigest()[:16],16)%100

def history_audit():
    d=load_data();panel=pd.read_pickle(PRIV/'panel.pkl')
    train=d[d.wave.le(20)];valid=d[d.wave.between(21,23)];test=d[d.wave.between(24,26)].copy()
    eligible=set(train.pid);observed=set(panel.loc[panel.wave.le(20),'pid'])
    a=train[train.pid.map(group).lt(70)];b=valid[valid.pid.map(group).between(70,84)]
    fitted=joblib.load(PRIV/'models/person_disjoint_refit.joblib')
    pv=predict(fitted,b);raw=predict(fitted,test);maps=calibrators(pv,b.exit_label_t1)
    test['arm']=np.where(test.pid.map(group).lt(70),'training',np.where(test.pid.map(group).ge(85),'held_out','validation'))
    test['eligible_training_history']=test.pid.isin(eligible)
    test['any_panel_history']=test.pid.isin(observed)
    test['actually_used_in_fit']=test.pid.isin(set(a.pid))
    assert np.array_equal(test.actually_used_in_fit,test.arm.eq('training')&test.eligible_training_history)
    assert not test.loc[test.arm.eq('held_out'),'actually_used_in_fit'].any()
    test['raw']=raw;test['platt']=calibrated(maps,raw,'platt')
    cuts={m:f1_threshold(b.exit_label_t1,calibrated(maps,pv,m)) for m in ['raw','platt']}
    rows=[]
    for definition in ['all','eligible_training_history','any_panel_history']:
        levels=[None] if definition=='all' else [True,False]
        for history in levels:
            for arm in ['training','held_out']:
                mask=test.arm.eq(arm)
                if history is not None:mask &= test[definition].eq(history)
                frame=test[mask]
                for method in ['raw','platt']:
                    rows.append({'history_definition':definition,'history':history,'arm':arm,'method':method,'people':frame.pid.nunique(),'actually_used_people':frame.loc[frame.actually_used_in_fit,'pid'].nunique(),**metrics(frame.exit_label_t1,frame[method],cuts[method])})
    result=pd.DataFrame(rows);save(result,'disjoint_history_groups.csv')
    # Retain the four-cell disjoint partition within every replicate, with
    # person multiplicities shared between output variants and contrasts.
    draws=[];points=[]
    for definition in ['eligible_training_history','any_panel_history']:
        cells={};point={}
        for history in [True,False]:
            for arm in ['training','held_out']:
                frame=test[test.arm.eq(arm)&test[definition].eq(history)]
                ids,code=np.unique(frame.pid,return_inverse=True)
                scores={m:FastScore(frame.exit_label_t1,frame[m]) for m in ['raw','platt']}
                cells[(history,arm)]=(ids,code,scores)
                point[(history,arm)]={m:s.score(np.ones(len(frame)),False) for m,s in scores.items()}
        for method in ['raw','platt']:
            for metric in ['roc_auc','pr_auc','brier']:
                deltas={h:point[(h,'training')][method][metric]-point[(h,'held_out')][method][metric] for h in [True,False]}
                for name,value in [('eligible_or_observed_history',deltas[True]),('without_history',deltas[False]),('history_minus_no_history_gap',deltas[True]-deltas[False])]:
                    points.append({'history_definition':definition,'contrast':name,'method':method,'metric':metric,'estimate':value})
        for rep in range(500):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,147 if definition=='eligible_training_history' else 148,rep]));values={}
            for key,(ids,code,scores) in cells.items():
                w=rng.multinomial(len(ids),np.ones(len(ids))/len(ids))[code].astype(float)
                values[key]={m:s.score(w,False) for m,s in scores.items()}
            for method in ['raw','platt']:
                for metric in ['roc_auc','pr_auc','brier']:
                    delta={h:values[(h,'training')][method][metric]-values[(h,'held_out')][method][metric] for h in [True,False]}
                    for name,value in [('eligible_or_observed_history',delta[True]),('without_history',delta[False]),('history_minus_no_history_gap',delta[True]-delta[False])]:
                        draws.append({'history_definition':definition,'replicate':rep,'contrast':name,'method':method,'metric':metric,'difference':value})
    draws=pd.DataFrame(draws);save(draws,'disjoint_history_bootstrap_replicates.csv')
    out=[]
    for point in points:
        subset=draws
        for key in ['history_definition','contrast','method','metric']:subset=subset[subset[key].eq(point[key])]
        lo,hi=subset.difference.quantile([.025,.975]);out.append({**point,'lower':lo,'upper':hi,'resamples':500,'unit':'person within history-by-assignment cell'})
    save(pd.DataFrame(out),'disjoint_history_contrasts.csv')
    # Independent joins and reconstruction checks; only aggregate records are public.
    cross=test.groupby(['arm','eligible_training_history','any_panel_history'],dropna=False).agg(rows=('pid','size'),people=('pid','nunique'),events=('exit_label_t1','sum')).reset_index()
    save(cross,'disjoint_history_definition_audit.csv')
    test[['pid','wave','exit_label_t1','arm','eligible_training_history','any_panel_history','actually_used_in_fit','raw','platt']].to_pickle(PRIV/'disjoint_history_predictions.pkl')
    old=pd.read_csv(RES/'disjoint_same_model_groups.csv')
    for label,arm in [('hash_train_70_all_test_rows','training'),('hash_test_15_all_test_rows','held_out')]:
        for method in ['raw','platt']:
            x=old[old.evaluation_group.eq(label)&old.method.eq(method)].iloc[0]
            y=result[result.history_definition.eq('all')&result.arm.eq(arm)&result.method.eq(method)].iloc[0]
            assert all(np.isclose(x[k],y[k],atol=1e-12) for k in ['n','events','roc_auc','pr_auc','brier'])
    model_hash=hashlib.sha256((PRIV/'models/person_disjoint_refit.joblib').read_bytes()).hexdigest()
    dump({'status':'PASS','training_wave_range':[1,20],'eligible_history':'At least one observed, labelled wage-employed baseline at waves 1–20, irrespective of hash arm','any_panel_history':'At least one observed person-file interview at waves 1–20, including non-wage or unlabelled records','training_people':int(a.pid.nunique()),'validation_people':int(b.pid.nunique()),'fit_sha256':model_hash,'thresholds':cuts,'resamples':500,'seed_components':[SEED,147,148],'resampling':'Independent person resampling within each of four assignment-by-history cells; every person retains all test rows; common draws across metrics and raw/Platt outputs','scope':'Conditional diagnostic of one fitted object and realised assignment. A finite-sample negative-control difference need not be zero. Pure causal overlap or redevelopment effects are not identified.','original_all_arm_metrics_reproduced':True},RES/'disjoint_history_design.json')
    print(result.query("method == 'raw' and history_definition != 'any_panel_history'")[['history_definition','history','arm','n','people','events','roc_auc','brier']].to_string(index=False),flush=True)
    print(pd.DataFrame(out).query("history_definition == 'eligible_training_history' and method == 'raw' and metric == 'roc_auc'").to_string(index=False),flush=True)

def numeric_audit():
    seeds=pd.read_csv(RES/'seed_metrics.csv');contrasts=pd.read_csv(RES/'seed_contrasts.csv');checks=[]
    for _,r in contrasts.iterrows():
        a=seeds[seeds.model.eq(r.model_A)&seeds.method.eq(r.method)].set_index('seed')[r.metric]
        b=seeds[seeds.model.eq(r.model_B)&seeds.method.eq(r.method)].set_index('seed')[r.metric]
        delta=a-b;assert len(delta)==10 and not delta.isna().any()
        actual=[delta.mean(),delta.std(ddof=1),delta.min(),delta.max(),(delta>0).sum(),(delta<0).sum()]
        expected=[r.mean_A_minus_B,r.sd_A_minus_B,r.minimum,r.maximum,r.positive_differences,r.negative_differences]
        assert np.allclose(actual,expected,atol=1e-12)
        checks.append({'contrast':r.model_A+' - '+r.model_B,'method':r.method,'metric':r.metric,'verified':True})
    save(pd.DataFrame(checks),'seed_contrast_recalculation.csv')
    sub=pd.read_csv(RES/'subgroup_metrics.csv');age=sub[sub.model.eq('catboost')&sub.method.eq('platt')&sub.subgroup_variable.eq('age_group')]
    old=age[age.group.eq('65plus')].iloc[0];selected=int(round((age.n*age.positive_rate).sum()));older=int(round(old.n*old.positive_rate));tp=int(round(old.events*old.recall))
    dump({'selected_all':selected,'selected_age65plus':older,'share_selected_age65plus':older/selected,'true_positives_age65plus':tp,'precision_age65plus':tp/older,'source':'subgroup_metrics.csv; CatBoost Platt common threshold'},RES/'age65_selection_audit.json')
    assert selected==6522 and older==2635
    print('Seed contrasts independently recalculated:',len(checks));print('Older selected share / precision:',older/selected,tp/older)

if __name__=='__main__':
    history_audit();numeric_audit()
