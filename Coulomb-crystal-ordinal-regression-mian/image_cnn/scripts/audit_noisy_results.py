import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
SEEDS=(42,12345,2026,43,44)
METHODS=('ordinal_regression','pure_classification')
METRICS=('accuracy','mae','rmse','acc_plus_minus_1','max_abs_error','loss')

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--require-results',action='store_true');a=p.parse_args();root=a.root.resolve(); noisy=root/'data/noisy_gaussian_mu0_sigma0.1'; manifest=noisy/'split_manifest.csv'; rows=list(csv.DictReader(manifest.open()))
 counts={s:sum(r['split']==s for r in rows) for s in ('train','val','test')}; assert len(rows)==2880 and counts=={'train':1920,'val':480,'test':480}; assert len({r['path'] for r in rows})==2880; assert all((root/r['path']).is_file() and (root/r['path']).stat().st_size>0 for r in rows); assert all(100<=int(r['n'])<180 for r in rows)
 metadata=json.loads((noisy/'metadata.json').read_text()); assert metadata['mu']==0.0 and abs(metadata['sigma']-0.1)<1e-12 and abs(metadata['variance']-0.01)<1e-12
 result_root=root/'results/noisy_gaussian_mu0_sigma0.1'; records=[]
 for method in METHODS:
  for seed in SEEDS:
   d=result_root/method/f'seed{seed}'; summary=d/'summary.json'
   if not summary.is_file():
    if a.require_results: raise FileNotFoundError(summary)
    continue
   for required in ('history.json','validation_predictions.csv','test_predictions.csv'): assert (d/required).is_file(), d/required
   payload=json.loads(summary.read_text())
   for split in ('validation','test'):
    row={'method':method,'seed':seed,'split':split}; row.update({m:payload[split][m] for m in METRICS}); records.append(row)
 if records:
  result_root.mkdir(parents=True,exist_ok=True)
  with (result_root/'per_seed_metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
  aggregate={}
  for method in METHODS:
   aggregate[method]={}
   for split in ('validation','test'):
    subset=[r for r in records if r['method']==method and r['split']==split]
    if len(subset)!=len(SEEDS):
     if a.require_results: raise RuntimeError(f'incomplete {method} {split}: {len(subset)}/5')
     continue
    aggregate[method][split]={m:{'mean':float(np.mean([r[m] for r in subset])),'std':float(np.std([r[m] for r in subset],ddof=1))} for m in METRICS}
  (result_root/'aggregate_summary.json').write_text(json.dumps({'noise':metadata,'seeds':list(SEEDS),'metrics':aggregate},indent=2))
 print(json.dumps({'manifest_sha256':sha256(manifest),'source_manifest_sha256':metadata['source_manifest_sha256'],'split_counts':counts,'completed_summaries':len(records)//2,'required_summaries':10,'metadata':metadata},indent=2))
if __name__=='__main__':main()
