"""Inventory downloaded released ground truth and validate the saved full-S29 pilot. No inference."""
import json,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]; source=root/'data/moscato-release'; rows=[]
for p in sorted((source/'annotations/ground_truth').rglob('*.json')):
 if '_idx' in p.stem:continue
 b=p.read_bytes()
 if b.startswith(b'version https://git-lfs'): rows.append({'file':str(p.relative_to(source)),'error':'LFS pointer'});continue
 d=json.loads(b); videos=[]
 for vid,a in d.items():
  states=a.get('state',{});videos.append({'video_id':vid,'objects':list(states),'annotation_lengths':sorted(set(len(t) for t in states.values()))})
 rows.append({'file':str(p.relative_to(source)),'sha256':hashlib.sha256(b).hexdigest(),'videos':videos,'video_count':len(videos),'video_object_pairs':sum(len(v['objects']) for v in videos)})
out=root/'runs/readiness-audit';out.mkdir(parents=True,exist_ok=True);(out/'dataset_inventory.json').write_text(json.dumps(rows,indent=2))
for r in rows: print(r['file'],r.get('video_count'),r.get('video_object_pairs'))
p=root/'runs/batches/full-video-pilot-v1/S29_Salad_7150991-1/model';m=json.loads((p/'feature_manifest.json').read_text());data=np.load(p/'raw_probabilities.npz');checks={k:{'shape':list(data[k].shape),'finite':bool(np.isfinite(data[k]).all()),'normalized':bool(np.allclose(data[k].sum(1),1,atol=1e-5))} for k in data.files}; print('Prior run:',checks,'seconds',m['elapsed_seconds']);(out/'prior_run_check.json').write_text(json.dumps({'checks':checks,'elapsed_seconds':m['elapsed_seconds'],'object_windows':m['object_windows'],'missing_object_windows':m['missing_object_windows'],'wandb_receipt_exists':(p/'wandb_upload_receipt.json').exists()},indent=2))
