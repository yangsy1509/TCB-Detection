"""Run the original RGB+CTH full-scene inference with portable paths."""
import argparse,os,runpy
from pathlib import Path
from release_common import ROOT

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-root',type=Path,required=True,help='Timestamp folders containing CHW RGB.npy and2D CTH.npy')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--normalization-dir',type=Path,default=None)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/scenes')
    p.add_argument('--gpu',default=None)
    args=p.parse_args(argv)
    for path in [args.scene_root,args.checkpoint]:
        if not path.exists():raise FileNotFoundError(path)
    os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
    if args.gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    os.environ.update(TCB_SCENE_ROOT=str(args.scene_root.resolve()),TCB_MODEL_DIR=str((args.normalization_dir or args.checkpoint.parent).resolve()),TCB_CHECKPOINT=str(args.checkpoint.resolve()),TCB_OUTPUT_DIR=str(args.output_dir.resolve()),TCB_MAMBAVISION_ROOT=str(ROOT/'third_party/MambaVision'))
    runpy.run_path(str(ROOT/'inference/predict_scenes_rgb_cth.py'),run_name='__main__')

if __name__=='__main__':main()
