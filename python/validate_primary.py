"""Meaningful data, information-boundary, metric, and reporting checks."""
from __future__ import annotations
import hashlib,importlib,json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from common import *
from feature_definitions import MODEL_FEATURES
from run_pipeline import validate_raw_manifest

checks=[]
def check(name,condition,detail):
    assert bool(condition),name+': '+str(detail)
    checks.append({'check':name,'status':'PASS','detail':detail})

def main():
    d=load_data();risk=pd.read_pickle(PRIV/'risk.pkl');panel=pd.read_pickle(PRIV/'panel.pkl');t=pd.read_pickle(PRIV/'test_predictions.pkl');v=pd.read_pickle(PRIV/'valid_predictions.pkl')
    check('unique_person_wave',not panel.duplicated(['pid','wave']).any(),len(panel))
    check('analysis_sample',len(d)==139566 and d.pid.nunique()==19339 and int(d.exit_label_t1.sum())==15171,{'rows':len(d),'people':d.pid.nunique(),'events':int(d.exit_label_t1.sum())})
    check('test_sample',len(t)==27063 and t.pid.nunique()==11187 and int(t.exit_label_t1.sum())==2395,{'rows':len(t),'people':t.pid.nunique(),'events':int(t.exit_label_t1.sum())})
    check('only_wage_baselines',d.state.eq('wage').all(),d.state.value_counts().to_dict())
    check('absent_followup_not_positive',risk.loc[~risk.outcome_observed,'exit_label_t1'].isna().all(),int((~risk.outcome_observed).sum()))
    check('administrative_vs_missing',int(risk.administrative_end.sum())==9556 and int((risk.wave.le(26)&~risk.outcome_observed).sum())==10980,'9556 administrative; 10980 eligible missing')
    q=panel.employment_status_raw.eq(2);check('questionnaire_precedes_last_job',panel.loc[q,'state'].eq('nonemployed').all(),int(q.sum()))
    xx=d[d.wave.between(24,26)&d.exit_label_t1.eq(1)&d.next_employee_status_raw.isna()];check('test_missing_detail_routing',len(xx)==2006 and xx.next_employment_status_raw.eq(2).all(),'2006 of 2395 test positives independently routed as nonemployment')
    # Reconstruct labels independently by an adjacent-wave panel lookup.
    nxt=panel[['pid','wave','state']].copy();nxt.wave=nxt.wave-1;test=d[['pid','wave','exit_label_t1']].merge(nxt,on=['pid','wave'],validate='one_to_one');check('adjacent_outcome_identity',np.array_equal(test.exit_label_t1.to_numpy(int),test.state.ne('wage').to_numpy(int)),len(test))
    cols=json.loads((RES/'model_features.json').read_text());forbidden={'pid','hhid','exit_label_t1','next_state','destination','has_next_wave','outcome_observed'};check('feature_allowlist',not(forbidden&set(cols)) and len(cols)==138 and set(cols).issubset(MODEL_FEATURES),'138 documented predictors; identifiers and future fields excluded')
    fold=pd.read_csv(RES/'forward_folds.csv');check('forward_oof_information_cutoff',(fold.latest_training_outcome_wave<fold.oof_start).all(),fold[['train_end','latest_training_outcome_wave','oof_start']].to_dict('records'))
    oof=pd.read_pickle(PRIV/'forward_oof.pkl');check('oof_cohort',oof.wave.between(9,20).all() and not oof.duplicated(['pid','wave']).any(),len(oof))
    # Independently verify historical wage counts for randomly selected analysis people.
    rng=np.random.default_rng(SEED);ids=rng.choice(d.pid.unique(),30,replace=False);base=panel[panel.pid.isin(ids)].sort_values(['pid','wave']).copy();base['prior']=base.groupby('pid').state.transform(lambda x:x.eq('wage').cumsum()-x.eq('wage').astype(int));z=d[d.pid.isin(ids)].merge(base[['pid','wave','prior']],on=['pid','wave']);check('history_counts_use_prior_states',np.allclose(z.prior,z.prior_wage_worker_count),len(z))
    # Person-disjoint assignment is deterministic and disjoint at all supervised stages.
    assign=d.pid.map(lambda p:int(hashlib.sha256(('qq20261026:'+str(int(p))).encode()).hexdigest()[:16],16)%100)
    a=d[d.wave.le(20)&assign.lt(70)];b=d[d.wave.between(21,23)&assign.between(70,84)];c=d[d.wave.between(24,26)&assign.ge(85)]
    check('genuine_person_disjoint_refit',not(set(a.pid)&set(b.pid) or set(a.pid)&set(c.pid) or set(b.pid)&set(c.pid)),[len(a),len(b),len(c)])
    check('disjoint_sample_counts',[len(a),len(b),len(c)]==[60262,3970,4015], [a.pid.nunique(),b.pid.nunique(),c.pid.nunique()])
    allscores=BASES+['forward_stack','forward_stack_balanced','shuffled_stack'];thresholds=pd.read_csv(RES/'selected_thresholds.csv');metricfile=pd.read_csv(RES/'model_calibration_metrics.csv');metricfile=metricfile[metricfile.partition.eq('test')]
    for n in allscores:
        for m in ['raw','platt','isotonic']:
            col=n if m=='raw' else n+'_'+m;p=t[col].to_numpy();check('finite_probability_'+col,np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),len(p))
            chosen=thresholds[(thresholds.model==n)&(thresholds.method==m)].threshold.iloc[0];check('validation_threshold_'+col,np.isclose(chosen,f1_threshold(v.exit_label_t1,v[col])),float(chosen))
            recomputed=metrics(t.exit_label_t1,p,chosen);r=metricfile[(metricfile.model==n)&(metricfile.method==m)].iloc[0]
            check('metric_source_'+col,all(np.isclose(recomputed[k],r[k],atol=1e-10) for k in ['roc_auc','pr_auc','brier','recall20','ece10','calibration_intercept','calibration_slope']),str(col))
    fast=importlib.import_module('cluster_inference').FastScore;y=rng.binomial(1,.2,1000);p=np.round(rng.uniform(0,.5,1000),2);weights=rng.integers(0,5,1000).astype(float);aa=metrics(y,p,weights=weights);bb=fast(y,p).score(weights);check('weighted_tied_metric_independent_validation',all(np.isclose(aa[k],bb[k],atol=1e-10) for k in bb),'Fast bootstrap metrics agree with sklearn/direct definitions, including ties and zero weights')
    ci=pd.read_csv(RES/'primary_cluster_confidence_intervals.csv');draw=pd.read_csv(RES/'aggregate_bootstrap_replicates.csv');check('bootstrap_replicates',draw.replicate.nunique()==1000 and draw.model.nunique()==15 and len(draw)==15000,{'replicates':1000,'models':15,'rows':15000})
    check('finite_primary_intervals',np.isfinite(ci[['estimate','lower','upper']]).all().all() and ci.lower.le(ci.upper).all(),len(ci))
    pairs=pd.read_csv(RES/'paired_cluster_model_contrasts.csv')
    for _,r in pairs.iterrows():
        da=draw[draw.model.eq(r.model_A)].set_index('replicate')[r.metric];db=draw[draw.model.eq(r.model_B)].set_index('replicate')[r.metric];lo,hi=(da-db).quantile([.025,.975]);assert np.isclose(lo,r.lower) and np.isclose(hi,r.upper)
    check('paired_contrasts_reproduce',True,len(pairs))
    util=pd.read_csv(RES/'utility_scenarios.csv');u=util.benefit_TP*util.test_TP-util.cost_FP*util.test_FP-util.cost_FN*util.test_FN;check('utility_arithmetic_single_source',np.array_equal(u,util.utility_units),len(util))
    check('utility_count_identity',(util.test_TP+util.test_FN).eq(2395).all(),len(util))
    iters=[]
    for path in (PRIV/'models').glob('*.joblib'):
        if 'logistic' not in path.stem or 'calibrators' in path.stem:continue
        fitted=joblib.load(path)
        if isinstance(fitted,dict) and 'model' in fitted:
            est=fitted['model']['estimator'];iters.append({'model':path.stem,'iterations':int(est.n_iter_[0]),'limit':int(est.max_iter)})
            assert est.n_iter_[0]<est.max_iter,path.name
    check('logistic_convergence',True,iters);save(pd.DataFrame(iters),'logistic_convergence.csv')
    audit=pd.read_csv(RES/'raw_file_audit.csv');check('core_raw_coverage',all((ROOT/'raw'/f'eklips{w:02d}{k}.xlsx').exists() for w in range(1,28) for k in ['p','h']),'All 54 required inputs available')
    source_hashes=validate_raw_manifest(ROOT)
    expected=pd.read_csv(ROOT/'config/raw_file_audit.csv').set_index('file');actual=audit.set_index('file')
    check('complete_reported_source_audit',actual.index.is_unique and expected.index.is_unique and len(actual)==len(expected) and set(actual.index)==set(expected.index) and actual.sha256.reindex(expected.index).equals(expected.sha256) and actual.read_status.eq('complete').all(),f'All {len(expected)} reported file identities match the declared source manifest and have complete read status')
    for _,r in audit.iterrows():
        path=ROOT/'raw'/r.file;assert path.stat().st_size==r.bytes;assert source_hashes[r.file]==r.sha256
    check('raw_checksums',True,{'fully_read_files':len(audit),'manifest_files':len(expected),'modelling_inputs':54,'work_history_audited':'eklips27w.xlsx' in source_hashes,'work_history_used_for_modelling':False})
    result={'status':'PASS','checks_passed':len(checks),'checks':checks,'scope':'Conditional fitted-model, observed-follow-up benchmark; source coverage follows the declared manifest; no full survey-design inference'};dump(result,RES/'validation_report.json');print(json.dumps({'status':'PASS','checks_passed':len(checks)}))

if __name__=='__main__':main()
