import argparse, subprocess, sys
from pathlib import Path
SEEDS=(42,12345,2026,43,44)
p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--method",choices=("ordinal","classification","both"),default="both"); p.add_argument("--epochs",type=int,default=200); p.add_argument("--workers",type=int,default=2); p.add_argument("--device",default="auto",choices=("auto","cpu","cuda")); p.add_argument("--weights",default="default",choices=("default","none")); a=p.parse_args(); root=a.root.resolve(); py=sys.executable
methods=("ordinal","classification") if a.method=="both" else (a.method,)
for method in methods:
 script=root/"scripts"/f"train_{method}.py"
 for seed in SEEDS:
  cmd=[py,str(script),"--root",str(root),"--seed",str(seed),"--epochs",str(a.epochs),"--workers",str(a.workers),"--device",a.device,"--weights",a.weights]; print("RUN",*cmd,flush=True); subprocess.run(cmd,check=True)
