"""Run the analysis pipeline sequentially with reproducibility guards.

Usage: python python/run_pipeline.py [--analysis-only | --figures-only]
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def validate_raw_manifest(root=ROOT):
    """Verify each declared source before any cached analysis can be reused.

    The manifest determines audit coverage, including unused sources when they
    are included in a full-folder audit. Raw records are read locally only.
    """
    root=Path(root)
    with (root/'config/raw_file_audit.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'file','bytes','sha256'}.issubset(reader.fieldnames or []):
            raise RuntimeError('Source manifest must contain file, bytes and sha256 columns.')
        expected=list(reader)
    names=[r['file'] for r in expected]
    if not names or len(names)!=len(set(names)):
        raise RuntimeError('Source manifest must contain unique, nonempty filenames.')
    if any(not n or Path(n).name!=n or '/' in n or '\\' in n or n in {'.','..'} for n in names):
        raise RuntimeError('Source manifest entries must be filenames within raw/.')
    core={f'eklips{w:02d}{k}.xlsx' for w in range(1,28) for k in ['p','h']}
    if not core.issubset(names):
        raise RuntimeError('Source manifest must include all 27 person and 27 household XLSX files.')
    absent=[n for n in names if not (root/'raw'/n).is_file()]
    if absent:
        raise RuntimeError(f'The reported source audit requires all {len(names)} manifest files. Missing: '+', '.join(absent))
    audited_files={p.name for p in (root/'raw').iterdir() if p.is_file() and
                   (p.suffix in {'.pdf','.xls'} or (p.suffix=='.xlsx' and p.name.startswith('eklips')))}
    unlisted=sorted(audited_files-set(names))
    if unlisted:
        raise RuntimeError('Audit inputs are absent from the source manifest: '+', '.join(unlisted)+'. Update the declared audit or move unrelated inputs out of raw/.')
    hashes={}
    for row in expected:
        path=root/'raw'/row['file'];expected_hash=row['sha256'].lower()
        if len(expected_hash)!=64 or any(c not in '0123456789abcdef' for c in expected_hash):
            raise RuntimeError('Invalid SHA-256 in source manifest: '+row['file'])
        try:expected_size=int(row['bytes'])
        except (TypeError,ValueError):
            raise RuntimeError('Invalid byte count in source manifest: '+row['file']) from None
        if path.stat().st_size!=expected_size:
            raise RuntimeError('Source byte count differs from the manifest: '+row['file'])
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        hashes[row['file']]=digest.hexdigest()
        if hashes[row['file']]!=expected_hash:
            raise RuntimeError('Source checksum differs from the manifest: '+row['file'])
    return hashes

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group();g.add_argument('--analysis-only',action='store_true');g.add_argument('--figures-only',action='store_true');args=ap.parse_args()
    for name in ['logs','results','private','figures','release']:(ROOT/name).mkdir(exist_ok=True)
    env=os.environ.copy();env.setdefault('OPENBLAS_NUM_THREADS','1');env.setdefault('OMP_NUM_THREADS','2');env.setdefault('MPLBACKEND','Agg')
    if args.figures_only:steps=[['make_figures.py']]
    else:
        # Prevent silent cache reuse with modified core files or analysis code.
        source_inputs=validate_raw_manifest(ROOT)
        core_names={f'eklips{w:02d}{k}.xlsx' for w in range(1,28) for k in ['p','h']}
        inputs={name:source_inputs[name] for name in sorted(core_names)}
        code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'python').glob('*.py')) if not p.name.startswith(('make_documents','report_','response_revision'))}
        signature={'source_inputs':source_inputs,'core_inputs':inputs,'analysis_code':code};guard=ROOT/'private'/'reproduction_signature.json'
        if guard.exists() and json.loads(guard.read_text())!=signature:raise RuntimeError('Raw inputs or analysis code changed. Move the existing private/ cache aside before a fresh full run; do not mix fits across versions.')
        guard.write_text(json.dumps(signature,indent=2))
        steps=[['prepare_data.py'],['sensitivity_analyses.py','audits'],['fit_models.py'],['sensitivity_analyses.py','models'],['sensitivity_analyses.py','predictions'],['cluster_inference.py','inference'],['cluster_inference.py','shap'],['extended_analyses.py','audit'],['extended_analyses.py','precision'],['extended_analyses.py','drift'],['extended_analyses.py','origins'],['extended_analyses.py','predictions'],['extended_analyses.py','logit'],['extended_analyses.py','household'],['seed_stability.py'],['history_stratified_audit.py'],['make_figures.py'],['validate_primary.py'],['validate_extensions.py'],['validate_review.py']]
    for step in steps:
        name='_'.join(step).replace('.py','');started=time.time();print('RUN',name,flush=True)
        with (ROOT/'logs'/(name+'.log')).open('w') as out:
            subprocess.run([sys.executable,str(ROOT/'python'/step[0]),*step[1:]],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
        print('DONE',name,round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__':main()
