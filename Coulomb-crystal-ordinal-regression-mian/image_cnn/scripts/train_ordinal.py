import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torchvision import models
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
from ioncount_portable.data import add_args,resolve_device,load_manifest,make_loaders,seed_everything
class OrdinalNet(nn.Module):
 def __init__(self,weights='default'):
  super().__init__(); w=models.ResNet18_Weights.DEFAULT if weights=='default' else None; n=models.resnet18(weights=w); o=n.conv1; n.conv1=nn.Conv2d(1,o.out_channels,o.kernel_size,o.stride,o.padding,bias=False)
  if w:
   with torch.no_grad(): n.conv1.weight.copy_(o.weight.mean(1,keepdim=True))
  n.fc=nn.Linear(n.fc.in_features,79); self.net=n
 def forward(self,x): return self.net(x)
def evaluate(model,loader,device,seed,split):
 model.eval(); out=[]; total=0;n=0
 with torch.no_grad():
  for x,y,t,paths in loader:
   logits=model(x.to(device)); yy=y.to(device); target=(torch.arange(79,device=device)[None,:]<yy[:,None]).float(); loss=nn.functional.binary_cross_entropy_with_logits(logits,target); pred=logits.sigmoid().gt(.5).sum(1); total+=loss.item()*len(y);n+=len(y)
   for i,p in enumerate(paths):
    true=int(y[i])+100; guess=int(pred[i])+100; e=guess-true; out.append({'seed':seed,'split':split,'sample_id':Path(p).name,'image_path':p,'N_true':true,'N_pred':guess,'error':e,'absolute_error':abs(e)})
 e=np.array([r['error'] for r in out]); a=np.abs(e); m={'count':len(e),'accuracy':float(np.mean(e==0)),'mae':float(np.mean(a)),'rmse':float(np.sqrt(np.mean(e*e))),'acc_plus_minus_1':float(np.mean(a<=1)),'max_abs_error':int(a.max()),'loss':total/n}; return out,m
def save_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def train(args):
 root=Path(args.root).resolve(); dev=resolve_device(args.device); out_root=(root/args.results_root).resolve() if args.results_root and not Path(args.results_root).is_absolute() else Path(args.results_root) if args.results_root else root/'results/ordinal_regression'; ck_root=(root/args.checkpoints_root).resolve() if args.checkpoints_root and not Path(args.checkpoints_root).is_absolute() else Path(args.checkpoints_root) if args.checkpoints_root else root/'checkpoints/ordinal_regression'; out=out_root/f'seed{args.seed}'; ck=ck_root/f'seed{args.seed}'; out.mkdir(parents=True,exist_ok=True);ck.mkdir(parents=True,exist_ok=True)
 manifest_path=Path(args.manifest) if args.manifest else root/'data/split_manifest.csv'; manifest_path=manifest_path if manifest_path.is_absolute() else root/manifest_path; metadata_path=manifest_path.parent/'metadata.json'; noise_metadata=json.loads(metadata_path.read_text()) if metadata_path.is_file() else None; seed_everything(args.seed); rows=load_manifest(root,args.manifest); ls=make_loaders(rows,args.workers,False); model=OrdinalNet(args.weights).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-6); best=-1;best_epoch=0;history=[]
 for ep in range(1,args.epochs+1):
  model.train(); total=0;n=0
  for x,y,_,_ in ls['train']:
   yy=y.to(dev); target=(torch.arange(79,device=dev)[None,:]<yy[:,None]).float();opt.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(x.to(dev)),target);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();total+=loss.item()*len(y);n+=len(y)
  sch.step(); _,vm=evaluate(model,ls['val'],dev,args.seed,'validation');
  if vm['accuracy']>best: best=vm['accuracy'];best_epoch=ep;torch.save(model.state_dict(),ck/'best.pt')
  history.append({'epoch':ep,'optimization_loss':total/n,'lr':sch.get_last_lr()[0],'best_epoch':best_epoch,**{f'validation_{k}':v for k,v in vm.items()}});(out/'history.json').write_text(json.dumps(history,indent=2))
 model.load_state_dict(torch.load(ck/'best.pt',map_location=dev));ls=make_loaders(rows,args.workers,True);vr,vm=evaluate(model,ls['val'],dev,args.seed,'validation');tr,tm=evaluate(model,ls['test'],dev,args.seed,'test');save_csv(out/'validation_predictions.csv',vr);save_csv(out/'test_predictions.csv',tr);(out/'summary.json').write_text(json.dumps({'model':'OrdinalRegression','variant':'independent_threshold_ordinal','seed':args.seed,'best_epoch':best_epoch,'validation':vm,'test':tm,'config':{'classes':80,'thresholds':79,'loss':'BCEWithLogitsLoss','optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'scheduler':'CosineAnnealingLR','eta_min':1e-6,'gradient_clip':1.0,'manifest':str(args.manifest or 'data/split_manifest.csv'),'results_root':str(out_root),'checkpoints_root':str(ck_root),'noise':noise_metadata}},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();add_args(p);train(p.parse_args())
