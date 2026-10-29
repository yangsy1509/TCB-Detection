"""Run one preserved paper-training script with portable paths."""
import argparse,json,os,runpy
from pathlib import Path
from release_common import ROOT,SCHEMES,read_manifest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scheme',choices=SCHEMES,default='RGB_CTH')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--catalog',type=Path,default=ROOT/'data/catalog.csv')
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--pretrained-checkpoint',type=Path,default=ROOT/'checkpoints/mambavision_tiny_1k.pth.tar')
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--sampling-seed',type=int,default=1004)
    p.add_argument('--training-seed',type=int,default=1004)
    p.add_argument('--gpu',default=None)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args(argv)
    if args.epochs<1:p.error('epochs must be positive')
    frame=read_manifest(args.catalog,args.data_root)
    print(frame.groupby(['year','label']).size().to_string())
    if args.check_only:
        print('Paths/catalog checked. No training or output.');return
    out=(args.output_dir or ROOT/'outputs'/args.scheme).resolve()
    os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
    if args.gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    import torch
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError('Use an NVIDIA CUDA PyTorch environment for MambaVision.')
    out.mkdir(parents=True,exist_ok=True)
    resolved=out/'resolved_catalog.csv'
    frame[['year','target','filename','rgb_path','ctt_path','cth_path']].to_csv(resolved,index=False)
    os.environ.update(TCB_CATALOG=str(resolved),TCB_OUTPUT_DIR=str(out),
        TCB_MAMBAVISION_ROOT=str(ROOT/'third_party/MambaVision'),
        TCB_PRETRAINED_CHECKPOINT=str(args.pretrained_checkpoint.resolve()),
        TCB_EPOCHS=str(args.epochs),TCB_SAMPLING_SEED=str(args.sampling_seed),TCB_TRAINING_SEED=str(args.training_seed))
    (out/'run_settings.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n',encoding='utf-8')
    runpy.run_path(str(ROOT/'training'/('train_'+args.scheme.lower()+'.py')),run_name='__main__')

if __name__=='__main__':main()
