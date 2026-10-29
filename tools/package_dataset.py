"""Losslessly package the full catalog into independent GitHub Release ZIP parts.
Raw arrays are copied byte-for-byte; no resampling, normalization or NaN changes.
The output directory is gitignored. Each part has<1.5GiB raw data by default.
"""
import argparse,csv,hashlib,json,os,sys,time,zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def atomic_json(path,data):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)

def collect_sources(catalog,source_root,max_bytes):
    source_root=Path(source_root).resolve();groups=[];seen=set();parents={}
    with open(catalog,encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            group=[]
            for kind in ['rgb','cth','ctt']:
                rel=Path(row[kind+'_path'])
                if rel.is_absolute() or '..' in rel.parts:raise ValueError(f'Not relative: {rel}')
                if rel.parent not in parents:
                    folder=(source_root/rel.parent).resolve()
                    if not folder.is_relative_to(source_root):raise ValueError('Path outside dataset')
                    with os.scandir(folder) as entries:
                        inventory={e.name:e.stat(follow_symlinks=False) for e in entries if e.is_file(follow_symlinks=False)}
                    parents[rel.parent]=(folder,inventory)
                folder,inventory=parents[rel.parent]
                src=folder/rel.name
                if rel.name not in inventory:raise FileNotFoundError(src)
                name=rel.as_posix()
                if name in seen:raise ValueError(f'Duplicate source: {name}')
                stat=inventory[rel.name]
                seen.add(name);group.append(dict(kind=kind,path=src,name=name,size=stat.st_size,mtime=stat.st_mtime_ns))
            groups.append(group)
    parts=[];part=[];size=0
    for group in groups:
        amount=sum(x['size'] for x in group)
        if amount>max_bytes:raise ValueError('One sample exceeds requested shard size')
        if part and size+amount>max_bytes:parts.append(part);part=[];size=0
        part.extend(group);size+=amount
    if part:parts.append(part)
    return parts

def pack_one(index,entries,out):
    name=f'tcb-dataset-part-{index:03d}.zip';dest=out/name;tmp=out/(name+'.tmp')
    checks={}
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for item in entries:
                arr=np.load(item['path'],mmap_mode='r',allow_pickle=False)
                shape=(512,512,3) if item['kind']=='rgb' else (512,512)
                if arr.shape!=shape or arr.dtype!=np.dtype('float32'):
                    raise ValueError(f'Unexpected array {item["name"]}: {arr.shape}/{arr.dtype}')
                arr._mmap.close();del arr
                h=hashlib.sha256()
                with open(item['path'],'rb') as source,archive.open(item['name'],'w',force_zip64=True) as target:
                    for chunk in iter(lambda:source.read(8*1024*1024),b''):
                        h.update(chunk);target.write(chunk)
                checks[item['name']]=h.hexdigest()
                stat=item['path'].stat()
                if stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime']:
                    raise ValueError(f'Source changed while packaging: {item["name"]}')
        if tmp.stat().st_size>=2*1024**3:raise ValueError('Part exceeds GitHub Release2GiB limit')
        os.replace(tmp,dest)
        return dict(name=name,bytes=dest.stat().st_size,sha256=file_hash(dest),files=checks)
    finally:
        if tmp.exists():tmp.unlink()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True,help='Dataset root containing RGB/,CTH/,CTT/')
    p.add_argument('--catalog',type=Path,default=ROOT/'data/catalog.csv')
    p.add_argument('--output-dir',type=Path,default=ROOT/'release_assets')
    p.add_argument('--max-mib',type=int,default=1536)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--repository',default='yangsy1509/TCB-Detection')
    p.add_argument('--tag',default='v1.0.0')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    if not 1<=args.max_mib<=1800 or args.workers<1:p.error('max-mib1..1800; workers>=1')
    parts=collect_sources(args.catalog,args.source_root,args.max_mib*1024**2)
    count=sum(len(part) for part in parts);size=sum(x['size'] for part in parts for x in part)
    print(f'{count} NPY files / {size/1024**3:.3f}GiB / {len(parts)} ZIP parts',flush=True)
    if args.dry_run:return
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps([(i['name'],i['size'],i['mtime']) for part in parts for i in part]).encode()).hexdigest()
    path=out/'package_state.json'
    state=json.loads(path.read_text()) if path.exists() else dict(source_signature=signature,max_mib=args.max_mib,parts={})
    if state['source_signature']!=signature or state['max_mib']!=args.max_mib:raise ValueError('Source/part settings changed; use a new output directory')
    pending=[]
    for i,part in enumerate(parts,1):
        old=state['parts'].get(str(i));dest=out/old['name'] if old else None
        if old and dest.is_file() and dest.stat().st_size==old['bytes'] and file_hash(dest)==old['sha256']:
            print(f'[Existing] {old["name"]}',flush=True)
        else:pending.append((i,part))
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(pack_one,i,part,out):i for i,part in pending}
        for future in as_completed(futures):
            i=futures[future];state['parts'][str(i)]=future.result();atomic_json(path,state)
            print(f'[Done{len(state["parts"])}/{len(parts)}] {state["parts"][str(i)]["name"]} | {(time.monotonic()-start)/60:.1f}min',flush=True)
    assets=[];checks={}
    for i in range(1,len(parts)+1):
        result=state['parts'][str(i)];checks.update(result['files'])
        assets.append({k:result[k] for k in ['name','bytes','sha256']})
        assets[-1]['url']=f'https://github.com/{args.repository}/releases/download/{args.tag}/{result["name"]}'
    checksum=''.join(f'{digest}  {name}\n' for name,digest in sorted(checks.items()))
    (ROOT/'data/dataset_files.sha256').write_text(checksum,encoding='utf-8')
    atomic_json(ROOT/'data/download_manifest.json',dict(repository=args.repository,tag=args.tag,assets=assets,npy_files=count,raw_bytes=size,status='Upload the listed assets to this GitHub Release before announcing data availability.'))
    (out/'SHA256SUMS').write_text(''.join(f'{x["sha256"]}  {x["name"]}\n' for x in assets),encoding='utf-8')
    print(f'Complete. Upload ZIP files and SHA256SUMS from {out}; commit updated data/download_manifest.json and data/dataset_files.sha256.',flush=True)

if __name__=='__main__':main()
