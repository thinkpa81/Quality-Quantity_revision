"""Reconstruct the panel and adjudicate prospective employment states from KLIPS raw files.

Raw data remain local. All released outputs are aggregate.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from pypdf import PdfReader
import xlrd
from feature_construction import CONCEPT_RULES, build_core_features
from feature_definitions import MODEL_FEATURES, NUMERIC_FEATURES, FEATURE_GROUPS

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'; PRIVATE=ROOT/'private'
RES.mkdir(exist_ok=True); PRIVATE.mkdir(exist_ok=True)

def csv(df,name): df.to_csv(RES/name,index=False,encoding='utf-8-sig')

def read_one(path):
    p=Path(path); w=int(re.search(r'eklips(\d\d)',p.name).group(1)); typ=re.search(r'eklips\d\d([a-z])',p.name).group(1)
    cache=PRIVATE/'extraction_cache';cache.mkdir(exist_ok=True)
    cp=cache/(p.name+'.pkl')
    if cp.exists():
        cached=pickle.loads(cp.read_bytes())
        if cached[0]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest():return cached
    raw=pd.read_excel(p,engine='calamine')
    raw.columns=[str(c).strip() for c in raw.columns]
    audit={'file':p.name,'wave':w,'type':typ,'rows':len(raw),'columns':len(raw.columns),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'read_status':'complete'}
    if typ not in ['p','h']:
        cp.write_bytes(pickle.dumps((audit,None)));return audit,None
    # KLIPS nonresponse is coded negatively. Positive numeric values are retained.
    for c in raw.columns:
        if pd.api.types.is_numeric_dtype(raw[c]): raw[c]=raw[c].mask(raw[c].lt(0))
    out=pd.DataFrame(index=raw.index)
    for rule in CONCEPT_RULES:
        if rule.source_type!=typ:continue
        matched=[c for c in raw if re.search(rule.pattern,c,re.I)]
        if matched:out[rule.concept]=raw[matched].bfill(axis=1).iloc[:,0]
    hh=f'hhid{w:02d}'
    if hh in raw:out['hhid']=raw[hh]
    elif 'hhid' in raw:out['hhid']=raw.hhid
    if typ=='p':
        for suffix in ['0202','0203','0204','0205','2801','2802','2803','2804','2806']:
            c=f'p{w:02d}'+suffix
            if c in raw:out['routing_'+suffix]=raw[c]
        for c in ['sample98','sample09','sample18','orghid98','orghid09','orghid18','hwaveent','jobclass','jobnum','jobtype']:
            if c in raw:out[c]=raw[c]
        for suffix in ['orig98','orig09','orig18','9102','9103']:
            c=f'p{w:02d}{suffix}'
            if c in raw:out['person_'+suffix]=raw[c]
        for label,prefix in [('weight98','w'),('weight09','sw'),('weight18','nw')]:
            cs=[f'{prefix}{w:02d}p_c',f'{prefix}{w:02d}p']
            ls=[f'{prefix}{w:02d}p_l',f'{prefix}{w:02d}p']
            out[label+'_cross']=next((raw[c] for c in cs if c in raw),np.nan)
            out[label+'_long']=next((raw[c] for c in ls if c in raw),np.nan)
    out['wave']=w;out['survey_year']=1997+w;out['source_file']=p.name
    cp.write_bytes(pickle.dumps((audit,out)))
    return audit,out

def classify_state(df):
    q=pd.to_numeric(df.employment_status_raw,errors='coerce')
    s=pd.to_numeric(df.employee_status_raw,errors='coerce')
    t=pd.to_numeric(df.employment_type_raw,errors='coerce')
    state=pd.Series('unresolved',index=df.index,dtype='str')
    state.loc[q.eq(2)]='nonemployed'
    state.loc[q.eq(1)&s.isin([1,2,3])]='wage'
    state.loc[q.eq(1)&s.eq(4)]='self_employed'
    state.loc[q.eq(1)&s.eq(5)]='unpaid_family'
    fallback=q.eq(1)&~s.isin([1,2,3,4,5])
    for value,label in [(1,'wage'),(2,'self_employed'),(3,'unpaid_family')]:state.loc[fallback&t.eq(value)]=label
    return state,fallback&t.isin([1,2,3])

def write_metadata():
    csv(pd.DataFrame([{'concept':r.concept,'source_type':r.source_type,'source_pattern':r.pattern,'description':r.description} for r in CONCEPT_RULES]),'source_variable_mapping.csv')
    wb=xlrd.open_workbook(ROOT/'raw'/'codebook1(Wave 1-27).xls');sheet=wb.sheet_by_name('Individual');out=[];var='';label=''
    selected=['p__0201','p__0203','p__0204','p__0205','p__0211','p__0314','p__2801','p__2802','p__2804','sample98','sample09','sample18']
    for i in range(sheet.nrows):
        row=sheet.row_values(i)
        if str(row[2]).strip():var=str(row[2]).strip();label=str(row[1]).strip()
        if var in selected:out.append({'sheet':'Individual','excel_row':i+1,'variable':var,'label':label,'value_label':row[3]})
    csv(pd.DataFrame(out),'outcome_codebook_extract.csv')

def build():
    write_metadata()
    paths=sorted((ROOT/'raw').glob('eklips*.xlsx'))
    audits=[];persons=[];households=[]
    with ProcessPoolExecutor(max_workers=3) as ex:
        for audit,frame in ex.map(read_one,map(str,paths)):
            audits.append(audit)
            if frame is not None:(persons if audit['type']=='p' else households).append(frame)
            print('READ',audit['file'],audit['rows'],audit['columns'],flush=True)
    for p in sorted((ROOT/'raw').glob('*')):
        if p.suffix=='.pdf':
            reader=PdfReader(p)
            txt='\n'.join(f'PDF page {i+1}\n'+(page.extract_text() or '') for i,page in enumerate(reader.pages))
            (PRIVATE/(p.stem+'.txt')).write_text(txt)
            audits.append({'file':p.name,'pages':len(reader.pages),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'read_status':'complete'})
        elif p.suffix=='.xls':
            wb=xlrd.open_workbook(p)
            for s in wb.sheets():
                # Read the complete sheet; codebook excerpts below retain relevant definitions.
                _=[s.row_values(i) for i in range(s.nrows)]
            audits.append({'file':p.name,'sheets':len(wb.sheets()),'rows':sum(s.nrows for s in wb.sheets()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'read_status':'complete'})
    csv(pd.DataFrame(audits),'raw_file_audit.csv')
    p=pd.concat(persons,ignore_index=True);h=pd.concat(households,ignore_index=True)
    assert not p.duplicated(['pid','wave']).any(),'Duplicate person-wave keys'
    csv(h[h.duplicated(['hhid','wave'],keep=False)].groupby('wave').agg(rows=('wave','size'),missing_key=('hhid',lambda s:s.isna().sum()),unique_keys=('hhid','nunique')).reset_index(),'household_duplicate_audit.csv')
    h=h[h.hhid.notna() & h.hhid.gt(0)].copy()
    assert not h.duplicated(['hhid','wave']).any(),'Duplicate positive household-wave keys'
    keep=[c for c in h if c not in p or c in ['hhid','wave']]
    panel=p.merge(h[keep],on=['hhid','wave'],how='left',validate='many_to_one',indicator=True)
    csv(panel.groupby(['wave','_merge'],observed=True).size().reset_index(name='n'),'household_join_audit.csv')
    panel=panel.drop(columns='_merge').sort_values(['pid','wave']).reset_index(drop=True)
    panel['state'],panel['state_fallback']=classify_state(panel)
    panel['first_observed_wave']=panel.groupby('pid').wave.transform('min')
    panel['cohort']=np.select([panel.sample98.isin([1,2]),panel.sample09.isin([1,2]),panel.sample18.isin([1,2])],['98_origin_or_branch','09_refreshment_or_branch','18_refreshment_or_branch'],default='other_or_unclassified')
    panel.to_pickle(PRIVATE/'panel.pkl')
    core=build_core_features(panel)
    # Rebuild history counters using observed states; do not count unknown as non-wage.
    sg=core.sort_values(['pid','wave']).copy(); sg['is_wage_worker_t']=sg.state.eq('wage').astype(float).where(sg.state.ne('unresolved'))
    sg['is_non_wage_worker_t']=sg.state.isin(['self_employed','unpaid_family']).astype(float).where(sg.state.ne('unresolved'))
    for col in ['is_wage_worker_t','is_non_wage_worker_t']:
        prior=sg.groupby('pid')[col].shift(1).fillna(0)
        dest='prior_wage_worker_count' if col=='is_wage_worker_t' else 'prior_non_wage_worker_count'
        if dest=='prior_non_wage_worker_count': prior=sg.groupby('pid')['state'].shift(1).isin(['self_employed','unpaid_family','nonemployed']).astype(float)
        sg[dest]=prior.groupby(sg.pid).cumsum()
    known_prior=(sg.groupby('pid').state.shift(1).isin(['wage','self_employed','unpaid_family','nonemployed']).astype(float)).groupby(sg.pid).cumsum()
    sg['prior_wage_worker_share']=sg.prior_wage_worker_count/known_prior.replace(0,np.nan)
    sg['age_final']=sg.age_final # Retain source birth-year-based age definition for continuity.
    # Align each baseline with the immediately following survey wave, never the next observed nonadjacent wave.
    follow=sg[['pid','wave','state','state_fallback','employment_status_raw','employment_type_raw','employee_status_raw']].copy()
    follow['wave']=follow.wave-1
    follow=follow.rename(columns={c:'next_'+c for c in follow if c not in ['pid','wave']})
    risk=sg[sg.state.eq('wage')].merge(follow,on=['pid','wave'],how='left',validate='one_to_one')
    risk['administrative_end']=risk.wave.eq(27)
    risk['has_next_wave']=risk.next_state.notna()
    risk['outcome_observed']=risk.has_next_wave & risk.next_state.ne('unresolved') & ~risk.administrative_end
    risk['exit_label_t1']=np.where(risk.outcome_observed,risk.next_state.ne('wage').astype(float),np.nan)
    risk['split']=np.select([risk.wave.le(20),risk.wave.le(23),risk.wave.le(26)],['train','validation','test'],default='administrative_end')
    risk['destination']=risk.next_state.fillna('no_followup')
    risk.to_pickle(PRIVATE/'risk.pkl')
    ana=risk[risk.outcome_observed].copy().replace([np.inf,-np.inf],np.nan)
    # Dedicated metadata, future states and identifiers are never in MODEL_FEATURES.
    ana.to_pickle(PRIVATE/'analysis.pkl')
    summary=risk.groupby(['split','wave'],observed=True).agg(baseline_n=('pid','size'),observed_n=('outcome_observed','sum'),next_wave_n=('has_next_wave','sum'),events=('exit_label_t1','sum'),unique_people=('pid','nunique')).reset_index()
    summary['unobserved_n']=summary.baseline_n-summary.observed_n;summary['year']=summary.wave+1997
    csv(summary,'sample_flow_by_wave.csv')
    csv(ana.groupby('split').agg(n=('pid','size'),individuals=('pid','nunique'),events=('exit_label_t1','sum'),event_rate=('exit_label_t1','mean')).reset_index(),'sample_by_split.csv')
    csv(ana.groupby(['split','wave','destination']).size().reset_index(name='n'),'destination_by_wave.csv')
    csv(risk.groupby(['split','cohort'],observed=True).agg(baseline_n=('pid','size'),observed_n=('outcome_observed','sum'),events=('exit_label_t1','sum')).reset_index(),'cohort_distribution.csv')
    # Directly resolve the submitted positive class, including its missing work-status category.
    old=panel.sort_values(['pid','wave']).copy();g=old.groupby('pid');old['nextwave']=g.wave.shift(-1)
    for c in ['state','employment_status_raw','employment_type_raw','employee_status_raw']:old['next_'+c]=g[c].shift(-1)
    old=old[old.employee_status_raw.isin([1,2,3]) & old.nextwave.eq(old.wave+1)].copy()
    old['old_label']=(~old.next_employee_status_raw.isin([1,2,3])).astype(int)
    old['status_missing']=old.next_employee_status_raw.isna()
    old['split']=np.select([old.wave.le(20),old.wave.le(23)],['train','validation'],default='test')
    csv(old[old.old_label.eq(1)].groupby(['split','wave','status_missing','next_employment_status_raw','next_state'],dropna=False).size().reset_index(name='n'),'submitted_positive_class_audit.csv')
    joined=old[['pid','wave','old_label']].merge(ana[['pid','wave','exit_label_t1']],on=['pid','wave'],how='outer',indicator=True)
    csv(joined.groupby(['_merge','old_label','exit_label_t1'],observed=True,dropna=False).size().reset_index(name='n'),'submitted_revised_label_reconciliation.csv')
    features=[]
    train=ana[ana.wave.le(20)]
    for group,cols in FEATURE_GROUPS.items():
        for c in cols:
            features.append({'feature':c,'block':group,'type':'numeric' if c in NUMERIC_FEATURES else 'categorical','train_observed_n':int(train[c].notna().sum()) if c in train else 0,'selected':bool(c in train and train[c].notna().sum()>=20)})
    csv(pd.DataFrame(features),'feature_registry.csv')
    summary={'panel_rows':len(panel),'panel_people':int(panel.pid.nunique()),'risk_all_waves':len(risk),'risk_eligible':int(risk.wave.le(26).sum()),'administrative_end_rows':int(risk.administrative_end.sum()),'analysis_rows':len(ana),'analysis_people':int(ana.pid.nunique()),'events':int(ana.exit_label_t1.sum()),'unobserved_eligible':int((risk.wave.le(26)&~risk.outcome_observed).sum()),'raw_files_read':len(audits),'unknown_followup_rows':int((risk.has_next_wave&risk.next_state.eq('unresolved')&risk.wave.le(26)).sum())}
    (RES/'data_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__': build()
