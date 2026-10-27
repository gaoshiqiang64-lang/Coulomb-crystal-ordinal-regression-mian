import argparse, subprocess, sys
from pathlib import Path
SEEDS=(42,12345,2026,43,44)
p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--manifest',required=True); p.add_argument('--results-root',type=Path,required=True); p.add_argument('--checkpoints-root',type=Path,required=True); p.add_argument('--method',choices=('ordinal','classification','both'),default='both'); p.add_argument('--epochs',type=int,default=200); p.add_argument('--workers',type=int,default=2); p.add_argument('--device',default='auto',choices=('auto','cpu','cuda')); p.add_argument('--weights',default='default',choices=('default','none')); a=p.parse_args(); root=a.root.resolve(); methods=('ordinal','classification') if a.method=='both' else (a.method,)
for method in methods:
 output_name={'ordinal':'ordinal_regression','classification':'pure_classification'}[method]
 for seed in SEEDS:
  script=root/'scripts'/f'train_{method}.py'; cmd=[sys.executable,str(script),'--root',str(root),'--manifest',str(a.manifest),'--results-root',str(a.results_root/output_name), '--checkpoints-root',str(a.checkpoints_root/output_name),'--seed',str(seed),'--epochs',str(a.epochs),'--workers',str(a.workers),'--device',a.device,'--weights',a.weights]; print('RUN',*cmd,flush=True); subprocess.run(cmd,check=True)
