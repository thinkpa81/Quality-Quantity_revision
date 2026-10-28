"""Independent consistency checks for additional analyses and source identity."""
from __future__ import annotations
import json,hashlib,importlib
import numpy as np
import pandas as pd
from common import *

def main():
    checks=[]
    def check(name,condition,detail):
        assert bool(condition),name+': '+str(detail)
        checks.append({'check':name,'status':'PASS','detail':detail})
    read=lambda n:pd.read_csv(RES/n)
    d=load_data();tr=d[d.wave.le(20)];te=d[d.wave.between(24,26)];t=pd.read_pickle(PRIV/'test_predictions.pkl');v=pd.read_pickle(PRIV/'valid_predictions.pkl')
    constant=[c for c in feature_names(tr) if tr[c].nunique(dropna=True)<=1]
    check('three_training_constant_flags',set(constant)=={'post_2018_workweek_reform','covid_period','post_covid_period'},constant)
    check('period_flags_are_baseline_year',np.array_equal(d.post_2018_workweek_reform,(d.survey_year>=2018).astype(float)) and np.array_equal(d.covid_period,d.survey_year.between(2020,2021).astype(float)) and np.array_equal(d.post_covid_period,(d.survey_year>=2022).astype(float)),'No future/outcome year defines a period flag')
    precision=read('calibration_precision_audit.csv').set_index('fit_and_input_dtype');calibrated_models=read('model_calibration_metrics.csv');xp=calibrated_models[(calibrated_models.model=='xgboost')&(calibrated_models.method=='isotonic')&(calibrated_models.partition=='test')].iloc[0]
    check('xgboost_calibration_float64_consistency',all(np.isclose(precision.loc['float64',k],xp[k],atol=1e-12) for k in ['roc_auc','pr_auc','brier','recall20']),'Current XGBoost isotonic summaries agree with explicit float64 refit; float32 comparator retained for provenance')
    extras=read('fold_extra_exclusions.csv');check('fold1_exact_extra_exclusions',set(extras.feature_excluded_beyond_primary)=={'job_sat_overall_work','voluntary_work_form'} and extras.train_end.eq(7).all(),extras.to_dict('records'))
    ds=read('drift_standardization.csv');st=read('drift_strata.csv');co=read('drift_decomposition.csv')
    for _,r in ds.iterrows():
        g=d[d['split'].eq(r.partition)];check('drift_denominator_'+r.partition,len(g)==r.n and np.isclose(g.exit_label_t1.mean(),r.raw_event_rate) and np.isclose(g.cohort.str.startswith('18').mean(),r.share_2018_cohort),int(r.n))
        q=st[(st.partition==r.partition)&st.common_support.eq(True)];check('standardization_'+r.partition,np.isclose(q.reference_weight.sum(),1) and np.isclose((q.reference_weight*q.event_rate).sum(),r.standardized_event_rate),'Recomputed from stratum counts and fixed reference')
    check('absent_cohort_rates_not_imputed',st.loc[st.stratum.str.contains('18_refreshment')&st.partition.eq('train'),'event_rate'].isna().all(),'2018 strata excluded from historical common support')
    for pair,g in co.groupby(['from_partition','to_partition']):
        check('decomposition_identity_'+'_'.join(pair),np.isclose(g.loc[g.component.ne('total_event_rate_change'),'estimate'].sum(),g.loc[g.component.eq('total_event_rate_change'),'estimate'].iloc[0],atol=1e-12),g.estimate.to_list())
    origin=read('all_model_origins.csv');fold=read('origin_forward_folds.csv')
    check('all_five_models_both_origins',len(origin)==30 and all(set(g.model)==set(BASES+['forward_stack']) for _,g in origin.groupby('train_end')),'2 origins × 5 learners × 3 output maps')
    check('earlier_forward_label_timing',(fold.latest_base_outcome<fold.oof_start).all() and (fold.oof_end<=fold.origin_train_end).all(),fold.to_dict('records'))
    for end in [14,17]:
        test=pd.read_pickle(PRIV/f'origin{end}_test.pkl')
        for name in BASES+['forward_stack']:
            z=origin[(origin.train_end==end)&(origin.model==name)&(origin.method=='raw')].iloc[0];mm=metrics(test.exit_label_t1,test[name]);check(f'origin_metric_{end}_{name}',all(np.isclose(mm[k],z[k],atol=1e-12) for k in ['roc_auc','pr_auc','brier']),len(test))
    seeds=read('seed_metrics.csv');contrasts=read('seed_contrasts.csv');check('ten_complete_seed_results',len(seeds)==120 and set(seeds.seed)==set(range(SEED,SEED+10)) and seeds.groupby('seed').size().eq(12).all(),'Full stochastic/OOF and paired ablation results for ten fixed seeds')
    check('seed_metrics_finite',np.isfinite(seeds[['roc_auc','pr_auc','brier','calibration_intercept','calibration_slope']]).all().all(),len(seeds))
    for _,r in contrasts.iterrows():
        a=seeds[(seeds.model==r.model_A)&(seeds.method==r.method)].set_index('seed')[r.metric];b=seeds[(seeds.model==r.model_B)&(seeds.method==r.method)].set_index('seed')[r.metric];delta=a-b
        assert np.allclose([delta.mean(),delta.std(ddof=1),delta.min(),delta.max()],[r.mean_A_minus_B,r.sd_A_minus_B,r.minimum,r.maximum],atol=1e-12)
    check('seed_contrasts_independent_arithmetic',True,len(contrasts))
    seed0=pd.read_pickle(PRIV/'seed_scores'/f'{SEED}.pkl');check('seed0_matches_primary_scores',np.allclose(seed0['test'].catboost,t.catboost,atol=1e-12) and np.allclose(seed0['test'].forward_stack,t.forward_stack,atol=1e-12),'Primary seed reused without selecting the best seed')
    seed_design=json.loads((RES/'seed_design.json').read_text());check('matched_model_threads',seed_design['model_threads']==3 and seed_design['same_threads_as_primary'],'Three threads for every primary and repeated stochastic fit')
    sens=read('sensitivity_models.csv')
    for name in ['no_year','no_calendar']:
        a=seeds[(seeds.seed==SEED)&(seeds.model==name)&(seeds.method=='raw')].iloc[0];b=sens[(sens.analysis==name)&(sens.method=='raw')].iloc[0]
        check('seed0_ablation_matches_'+name,all(np.isclose(a[k],b[k],atol=1e-12) for k in ['roc_auc','pr_auc','brier']),'Matched seed and execution threads reproduce primary ablation')
    oof=pd.read_pickle(PRIV/'forward_oof.pkl');convergence=[]
    for seed in range(SEED,SEED+10):
        cached=pd.read_pickle(PRIV/'seed_scores'/f'{seed}.pkl');meta=LogisticRegression(C=1,max_iter=2000).fit(cached['forward_oof'],oof.exit_label_t1.astype(int))
        assert meta.n_iter_[0]<meta.max_iter and np.allclose(meta.coef_[0],cached['meta_coefficients'],atol=1e-12)
        convergence.append({'seed':seed,'iterations':int(meta.n_iter_[0]),'limit':meta.max_iter,'coefficients_reproduced':True})
    save(pd.DataFrame(convergence),'seed_meta_convergence.csv');check('all_seed_meta_fits_converged',True,convergence)
    age=read('matched_below65.csv');check('matched_below65_rows',age.n.eq(22987).all() and age.events.eq(1713).all(),'All-age and restricted fits evaluate identical baseline-age subset')
    for name in ['catboost','forward_stack']:
        sel=t.age_final.lt(65);mm=metrics(t.loc[sel,'exit_label_t1'],t.loc[sel,name]);z=age[(age.fit_population=='all ages')&(age.model==name)&(age.method=='raw')].iloc[0];check('age_baseline_recomputed_'+name,all(np.isclose(mm[k],z[k]) for k in ['roc_auc','pr_auc','brier']),len(t[sel]))
    floor=json.loads((RES/'stack_floor.json').read_text());check('stack_floor_respected',t.forward_stack.min()>=floor['mathematical_minimum_over_unit_cube']-1e-12,floor)
    audit=json.loads((RES/'389_event_audit.json').read_text());check('distinct_389_event_sets',audit['disjoint_events']==389 and audit['nonwage_events']==389 and audit['identical_event_sets'] is False,audit)
    th=read('calibration_raw_thresholds.csv')
    for _,r in th.iterrows():
        a=r.platt_intercept;b=r.platt_slope;z=expit((logit(r.platt_threshold)-a)/b);check('raw_cutoff_'+r.calibration_window+'_'+r.model,np.isclose(z,r.raw_equivalent_threshold,atol=1e-12),r.raw_equivalent_threshold)
        vv=v if r.calibration_window=='all_21_23' else v[v.wave.le(22)] if r.calibration_window=='pre2020_21_22' else v[v.wave.eq(23)]
        cal=calibrators(vv[r.model],vv.exit_label_t1);cut=f1_threshold(vv.exit_label_t1,calibrated(cal,vv[r.model],'platt'));selected=calibrated(cal,t[r.model],'platt')>=cut;assert np.isclose(cut,r.platt_threshold) and selected.sum()==r.test_selected
    check('calibration_windows_independently_refitted',True,len(th))
    sub=read('subgroup_metrics.csv');sex=sub[(sub.model=='catboost')&(sub.method=='platt')&(sub.subgroup_variable=='gender')];check('precision_selection_prevalence_identity',np.allclose(sex.recall,sex.precision*sex.positive_rate/sex.event_rate),'Descriptive algebra, not a causal fairness explanation')
    # Independently reconstruct components and verify whole-person retention.
    labels={p:p for p in t.pid.unique()}
    def root(x):
        while labels[x]!=x:labels[x]=labels[labels[x]];x=labels[x]
        return x
    for _,g in t[t.hhid.notna()&t.hhid.gt(0)].groupby(['wave','hhid']):
        for p in g.pid.iloc[1:]:
            a,b=root(g.pid.iloc[0]),root(p);labels[b]=a
    comp=t.pid.map(root);design=json.loads((RES/'household_design.json').read_text());check('household_component_reconstruction',comp.nunique()==design['connected_components'] and t.assign(c=comp).groupby('pid').c.nunique().eq(1).all(),design)
    for stem in ['household','logit_stack','matched_below65']:
        draws=read(stem+'_bootstrap_replicates.csv');pairs=read('household_component_contrasts.csv' if stem=='household' else stem+'_contrasts.csv')
        for _,r in pairs.iterrows():
            aa=draws[draws.model.eq(r.model_A)].set_index('replicate')[r.metric];bb=draws[draws.model.eq(r.model_B)].set_index('replicate')[r.metric];lo,hi=(aa-bb).quantile([.025,.975]);assert np.allclose([lo,hi],[r.lower,r.upper],atol=1e-12)
        check(stem+'_paired_intervals',True,len(pairs))
    check('rounding_training_followup',f'{100*8024/94566:.2f}'=='8.49','8024 / 94566')
    result={'status':'PASS','checks_passed':len(checks),'checks':checks,'scope':'Additional exploratory analyses; safeguards are scoped and do not claim complete survey-design or redevelopment inference'};dump(result,RES/'extension_validation_report.json');print(json.dumps({'status':'PASS','checks_passed':len(checks)}))

if __name__=='__main__':main()
