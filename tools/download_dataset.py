"""Download release parts, verify SHA256 and safely extract all dataset NPYs."""
import argparse,hashlib,json,os,stat,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def extract(archive,dest,expected):
    dest=dest.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name=member.filename;path=(dest/name).resolve()
            if not path.is_relative_to(dest) or stat.S_ISLNK(member.external_attr>>16) or name not in expected:
                raise ValueError(f'Unexpected archive path: {name}')
            if path.exists():
                if file_hash(path)==expected[name]:continue
                raise FileExistsError(f'Existing data differ; choose another output folder: {path}')
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.download.tmp')
            try:
                h=hashlib.sha256()
                with z.open(member) as source,tmp.open('wb') as f:
                    for chunk in iter(lambda:source.read(8*1024*1024),b''):h.update(chunk);f.write(chunk)
                if h.hexdigest()!=expected[name]:raise ValueError(f'NPY checksum mismatch: {name}')
                os.replace(tmp,path)
            finally:
                if tmp.exists():tmp.unlink()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'dataset')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'release_assets/download_cache')
    args=p.parse_args(argv)
    manifest=json.loads((ROOT/'data/download_manifest.json').read_text())
    if not manifest['assets']:raise SystemExit('Release assets have not been prepared/published yet.')
    expected={}
    for line in (ROOT/'data/dataset_files.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);expected[name]=digest
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    for asset in manifest['assets']:
        name=asset['name']
        if Path(name).name!=name:raise ValueError('Invalid asset name')
        path=args.cache_dir/name
        if not path.is_file() or file_hash(path)!=asset['sha256']:
            tmp=path.with_name(name+'.download.tmp')
            request=urllib.request.Request(asset['url'],headers={'User-Agent':'TCB-Detection-data-download'})
            try:
                with urllib.request.urlopen(request) as source,tmp.open('wb') as target:
                    for chunk in iter(lambda:source.read(8*1024*1024),b''):target.write(chunk)
                if file_hash(tmp)!=asset['sha256']:raise ValueError(f'ZIP checksum mismatch: {name}')
                os.replace(tmp,path)
            finally:
                if tmp.exists():tmp.unlink()
        extract(path,args.output_dir,expected);print(f'[Verified] {name}',flush=True)
    missing=[name for name in expected if not (args.output_dir/name).is_file()]
    if missing:raise ValueError(f'Missing extracted files: {missing[:5]}')
    print(f'Complete: {len(expected)} NPY files in {args.output_dir.resolve()}')

if __name__=='__main__':main()
