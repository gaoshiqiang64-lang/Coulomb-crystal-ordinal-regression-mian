import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

DEFAULT_SEED = 20260921
SIGMA = 0.1

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--seed',type=int,default=DEFAULT_SEED)
    p.add_argument('--force',action='store_true')
    a=p.parse_args(); root=a.root.resolve(); out=(a.output or root/'data/noisy_gaussian_mu0_sigma0.1').resolve()
    if out.exists() and any(out.iterdir()) and not a.force: raise SystemExit(f'output is non-empty: {out}; use --force to replace')
    if a.force and out.exists():
        import shutil; shutil.rmtree(out)
    image_dir=out/'images'; image_dir.mkdir(parents=True)
    manifest=root/'data/split_manifest.csv'; rows=list(csv.DictReader(manifest.open(newline='',encoding='utf-8')))
    rng=np.random.default_rng(a.seed); out_rows=[]; noise_stats=[]
    for i,row in enumerate(rows):
        source=root/row['path']; x=np.asarray(Image.open(source).convert('L'),dtype=np.float32)/255.0
        noise=rng.normal(0.0,SIGMA,size=x.shape).astype(np.float32); y=np.clip(x+noise,0.0,1.0).astype(np.float32)
        stored_rel=Path('images')/(Path(row['path']).stem+'.npy'); np.save(out/stored_rel,y,allow_pickle=False); rel=out.relative_to(root)/stored_rel
        nr=dict(row); nr['path']=str(rel).replace(chr(92),'/'); out_rows.append(nr)
        noise_stats.append(noise.ravel())
    with (out/'split_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(out_rows)
    n=np.concatenate(noise_stats)
    metadata={'source_manifest':str(manifest),'source_manifest_sha256':sha256(manifest),'noise_seed':a.seed,'mu':0.0,'sigma':SIGMA,'variance':SIGMA**2,'intensity_scale':'float grayscale [0,1]','operation':'clip(image + N(mu,sigma), 0, 1)','stored_dtype':'float32','image_count':len(rows),'split_counts':{s:sum(r['split']==s for r in rows) for s in ('train','val','test')},'preclip_noise_mean':float(n.mean()),'preclip_noise_variance':float(n.var())}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
if __name__=='__main__': main()
