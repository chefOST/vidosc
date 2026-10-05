"""Download selected CMU source archives; extract only exact video filenames."""
import hashlib
import json
import re
import shutil
import urllib.request
import urllib.parse
import zipfile
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from protocol import CONFIG, DATA, read, write, WORKSPACE

def main(config_path=CONFIG,manifest='sources.json'):
    cfg=read(config_path)
    DATA.mkdir(parents=True,exist_ok=True)
    page=urllib.request.urlopen('http://kitchen.cs.cmu.edu/main.php',timeout=60).read().decode(errors='replace')
    links=re.findall(r'href=[\"\x27]([^\"\x27]+)',page)
    def one(video):
        dest=DATA/(video+'.avi')
        for old in (WORKSPACE/'work/shray-multistate'/(video+'.avi'),WORKSPACE/'work/chefost-vidosc/data/moscato/videos'/(video+'.avi')):
            if old.exists() and not dest.exists(): shutil.copyfile(old,dest)
        subject,recipe=video.split('_')[:2]
        matches=[h for h in links if h.lower().endswith((subject+'_'+recipe+'_Video.zip').lower())]
        if len(matches)!=1: raise ValueError((video,matches))
        url=urllib.parse.urljoin('http://kitchen.cs.cmu.edu/main.php',matches[0])
        archive=DATA/(subject+'_'+recipe+'_Video.zip')
        if not dest.exists():
            if not archive.exists():
                print('Downloading',video,url,flush=True)
                partial=archive.with_suffix('.partial')
                with urllib.request.urlopen(url,timeout=180) as src,partial.open('wb') as out: shutil.copyfileobj(src,out)
                partial.replace(archive)
            with zipfile.ZipFile(archive) as z:
                names=[n for n in z.namelist() if Path(n).name.lower()==(video+'.avi').lower()]
                if len(names)!=1: raise ValueError((video,names))
                partial=dest.with_suffix('.partial')
                with z.open(names[0]) as src,partial.open('wb') as out: shutil.copyfileobj(src,out)
                partial.replace(dest)
        record={'video_id':video,'url':url,'video_bytes':dest.stat().st_size,'video_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        print('Ready',video,record['video_bytes'],flush=True)
        return record
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(one,cfg['train']+cfg['validation']+cfg['test']))
    write(DATA/manifest,records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',default=str(CONFIG));p.add_argument('--manifest',default='sources.json')
    args=p.parse_args();main(args.protocol,args.manifest)
