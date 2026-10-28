"""Independent numerical and source checks for the history-stratified analyses."""
from __future__ import annotations
import ast, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from common import ROOT, RES, PRIV, load_data, save, dump

def main():
    checks=[]
    def check(name, ok, detail):
        assert bool(ok), (name, detail)
        checks.append({'check':name,'status':'PASS','detail':detail})
    p=pd.read_pickle(PRIV/'disjoint_history_predictions.pkl')
    a=load_data(); panel=pd.read_pickle(PRIV/'panel.pkl')
    g=pd.read_csv(RES/'disjoint_history_groups.csv')
    c=pd.read_csv(RES/'disjoint_history_contrasts.csv')
    b=pd.read_csv(RES/'disjoint_history_bootstrap_replicates.csv')
    design=json.loads((RES/'disjoint_history_design.json').read_text())
    check('fixed_model_hash',hashlib.sha256((PRIV/'models/person_disjoint_refit.joblib').read_bytes()).hexdigest()==design['fit_sha256'],design['fit_sha256'])
    check('test_key_uniqueness',len(p)==27063 and not p.duplicated(['pid','wave']).any(),len(p))
    early=set(a.loc[a.wave.le(20),'pid']); anyearly=set(panel.loc[panel.wave.le(20),'pid'])
    check('eligibility_definition',p.eligible_training_history.eq(p.pid.isin(early)).all(),'Observed labelled wage-employed baseline at waves 1-20')
    check('alternative_interview_definition',p.any_panel_history.eq(p.pid.isin(anyearly)).all(),'Any person-file interview at waves 1-20')
    check('history_constant_within_person',p.groupby('pid')[['eligible_training_history','any_panel_history','arm']].nunique().eq(1).all().all(),'History and assignment remain person-level')
    check('actual_fit_membership',p.actually_used_in_fit.eq(p.arm.eq('training') & p.eligible_training_history).all(),'Only training-arm eligible people entered fitting')
    check('no_history_is_negative_control',not p.loc[~p.eligible_training_history,'actually_used_in_fit'].any(),'No person in either no-eligible-history arm entered supervised training')
    scores={'roc_auc':roc_auc_score,'pr_auc':average_precision_score,'brier':brier_score_loss}
    for i,r in g.iterrows():
        subset=p[p.arm.eq(r.arm)]
        if r.history_definition!='all':subset=subset[subset[r.history_definition].eq(r.history)]
        ok=(len(subset)==r.n and subset.pid.nunique()==r.people and subset.exit_label_t1.sum()==r.events)
        ok=ok and all(np.isclose(fun(subset.exit_label_t1,subset[r.method]),r[metric],atol=1e-12) for metric,fun in scores.items())
        check('group_metrics_'+str(i),ok,[r.history_definition,str(r.history),r.arm,r.method,int(r.n)])
    for definition in ['eligible_training_history','any_panel_history']:
        sub=g[g.history_definition.eq(definition)&g.method.eq('raw')]
        allarm=g[g.history_definition.eq('all')&g.method.eq('raw')].set_index('arm')
        for arm,z in sub.groupby('arm'):
            check('group_reconstruction_'+definition+'_'+arm,all(z[k].sum()==allarm.loc[arm,k] for k in ['n','people','events']),int(z.n.sum()))
    for i,r in c.iterrows():
        q=b
        for key in ['history_definition','contrast','method','metric']:q=q[q[key].eq(r[key])]
        z=g[g.history_definition.eq(r.history_definition)&g.method.eq(r.method)]
        delta={h:float(z[z.history.eq(h)&z.arm.eq('training')][r.metric].iloc[0]-z[z.history.eq(h)&z.arm.eq('held_out')][r.metric].iloc[0]) for h in [True,False]}
        estimate={'eligible_or_observed_history':delta[True],'without_history':delta[False],'history_minus_no_history_gap':delta[True]-delta[False]}[r.contrast]
        ci=q.difference.quantile([.025,.975]).values
        check('contrast_'+str(i),len(q)==500 and q.replicate.nunique()==500 and np.allclose([estimate,*ci],[r.estimate,r.lower,r.upper],atol=1e-12),[r.history_definition,r.contrast,r.method,r.metric])
    check('bootstrap_gap_algebra',np.allclose(b.pivot(index=['history_definition','replicate','method','metric'],columns='contrast',values='difference').eval('eligible_or_observed_history - without_history'),b.pivot(index=['history_definition','replicate','method','metric'],columns='contrast',values='difference')['history_minus_no_history_gap'],atol=1e-12),'Every draw preserves the difference between gaps')
    seeds=pd.read_csv(RES/'seed_metrics.csv'); contrasts=pd.read_csv(RES/'seed_contrasts.csv')
    for _,r in contrasts.iterrows():
        left=seeds[seeds.model.eq(r.model_A)&seeds.method.eq(r.method)].set_index('seed')[r.metric]
        right=seeds[seeds.model.eq(r.model_B)&seeds.method.eq(r.method)].set_index('seed')[r.metric]
        d=left-right
        assert len(d)==10 and np.allclose([d.mean(),d.std(ddof=1),d.min(),d.max(),(d>0).sum(),(d<0).sum()],[r.mean_A_minus_B,r.sd_A_minus_B,r.minimum,r.maximum,r.positive_differences,r.negative_differences],atol=1e-12)
    check('all_seed_contrast_recalculations',len(contrasts)==50,len(contrasts))
    age=json.loads((RES/'age65_selection_audit.json').read_text())
    check('age65_share_and_precision',age['selected_all']==6522 and age['selected_age65plus']==2635 and np.isclose(age['share_selected_age65plus'],2635/6522) and np.isclose(age['precision_age65plus'],490/2635),age)
    for path in (ROOT/'python').glob('*.py'):
        tree=ast.parse(path.read_text(),filename=path.name)
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module and re.match(r'^\d\d_',node.module):raise AssertionError((path,node.module))
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and re.fullmatch(r'\d\d_(?:prepare|models|sensitivities|inference|figures|validate|documents|v2_analyses|seed_stability|validate_v2)(?:\.py)?',node.value):raise AssertionError((path,node.value))
    check('descriptive_module_names',True,'All source files parse; no legacy numbered-module imports')
    result={'status':'PASS','checks_passed':len(checks),'inference_scope':'Conditional on the same fitted model and realised hash assignment; no causal overlap or redevelopment claim','checks':checks}
    dump(result,RES/'review_validation_report.json')
    print(json.dumps({'status':'PASS','checks_passed':len(checks)}))

if __name__=='__main__':main()
