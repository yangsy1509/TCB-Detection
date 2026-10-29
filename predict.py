"""Predict/evaluate labelled512x512 NPY patches for the four paper inputs."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from release_common import ROOT,SCHEMES,read_manifest,load_patch,normalize_patch

class PatchDataset:
    def __init__(self,frame,extras,mean,std):self.frame=frame;self.extras=extras;self.mean=mean;self.std=std
    def __len__(self):return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        return np.ascontiguousarray(normalize_patch(load_patch(row,self.extras),self.mean,self.std)),int(row.label),row['name']

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scheme',choices=SCHEMES,default='RGB_CTH')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/splits/test.csv')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--normalization-dir',type=Path,default=None)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--gpu',default=None)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args(argv)
    if args.batch_size<1 or args.workers<0:p.error('Invalid batch size/workers')
    frame=read_manifest(args.manifest,args.data_root)
    extras=SCHEMES[args.scheme];nc=3+len(extras)
    stats=args.normalization_dir or args.checkpoint.parent
    mean=np.load(stats/'train_channel_mean.npy').astype(np.float32)
    std=np.load(stats/'train_channel_std.npy').astype(np.float32)
    if mean.shape!=(nc,) or std.shape!=(nc,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Invalid saved TRAINING normalization')
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    # Verify a real input using the same transformations before loading the model.
    normalize_patch(load_patch(frame.iloc[0],extras),mean,std)
    if args.check_only:print(f'Checked{len(frame)} samples; no inference/output.');return
    os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
    if args.gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
    if not torch.cuda.is_available() or torch.version.cuda is None:raise RuntimeError('NVIDIA CUDA PyTorch is required')
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(ROOT/'third_party/MambaVision'))
    from mambavision.models import create_model
    model=create_model('mamba_vision_T',pretrained=False,num_classes=1,in_chans=nc,resolution=512)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if 'state_dict' in state:state=state['state_dict']
    model.load_state_dict(state,strict=True);model=model.cuda().eval()
    loader=DataLoader(PatchDataset(frame,extras,mean,std),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    rows=[]
    with torch.no_grad():
        for array,labels,names in loader:
            probability=torch.sigmoid(model(array.cuda(non_blocking=True))).view(-1)
            if not torch.isfinite(probability).all():raise ValueError('Non-finite prediction')
            for name,y,prob in zip(names,labels.tolist(),probability.cpu().tolist()):rows.append((name,int(y),float(prob),int(prob>.5)))
    result=pd.DataFrame(rows,columns=['name','label','probability','prediction'])
    if result.name.tolist()!=frame['name'].tolist():raise ValueError('Prediction order changed')
    output=args.output or ROOT/'outputs'/f'{args.scheme}_test_predictions.csv'
    output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output,index=False)
    y=result.label;p=result.prediction;prob=result.probability
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    metrics=dict(n=len(y),accuracy=float(accuracy_score(y,p)),TCB_precision=float(precision_score(y,p,zero_division=0)),TCB_recall=float(recall_score(y,p,zero_division=0)),TCB_F1=float(f1_score(y,p,zero_division=0)),AUROC=float(roc_auc_score(y,prob)) if y.nunique()==2 else None,AUPRC_AP=float(average_precision_score(y,prob)) if y.nunique()==2 else None,TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),threshold=.5,prediction_operator='>')
    output.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8');print(metrics)

if __name__=='__main__':main()
