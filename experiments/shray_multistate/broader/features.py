"""Frozen CLIP, official SlowFast, and optional genuine InternVideo feature extraction."""
import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from protocol import CONFIG,DATA,HERE,WORKSPACE,read,write,units
from prepare import load_upstream_encoder

FFMPEG=next((WORKSPACE/'work/ffmpeg').rglob('ffmpeg.exe'))
FFPROBE=FFMPEG.with_name('ffprobe.exe')

def video_info(video):
    path=DATA/(video+'.avi')
    info=json.loads(subprocess.check_output([str(FFPROBE),'-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,avg_frame_rate,width,height','-of','json',str(path)]))['streams'][0]
    fps=eval_fraction(info['avg_frame_rate'])
    count=int(info['nb_frames'])
    if count<=0 or fps<=0: raise ValueError('Invalid video metadata')
    return path,{'video_id':video,'raw_frame_count':count,'fps':fps,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def eval_fraction(value):
    a,b=value.split('/'); return float(a)/float(b)

def feature_path(video,backbone,smoke=False):
    return DATA/'features'/backbone/(video+('.smoke' if smoke else '')+'.pt')

def read_exact(pipe,n):
    data=bytearray()
    while len(data)<n:
        part=pipe.read(n-len(data))
        if not part: break
        data.extend(part)
    if not data: return None
    if len(data)!=n: raise ValueError('Partial decoded frame')
    return bytes(data)

def decoded_frames(path,side,step):
    filters=f'select=not(mod(n\\,{step})),scale=-2:{side},crop={side}:{side}'
    proc=subprocess.Popen([str(FFMPEG),'-v','error','-i',str(path),'-vf',filters,'-fps_mode','passthrough',
                           '-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        while True:
            frame=read_exact(proc.stdout,side*side*3)
            if frame is None: break
            yield np.frombuffer(frame,np.uint8).reshape(side,side,3).copy()
        error=proc.stderr.read().decode(errors='replace')
        if proc.wait()!=0: raise RuntimeError(error)
    finally:
        if proc.poll() is None: proc.terminate();proc.wait()
        proc.stdout.close();proc.stderr.close()

def video_windows(path,count,anchors):
    """32 fast-path frames spaced two raw frames apart; tail repeated, never beyond video."""
    stream=iter(decoded_frames(path,256,2))
    total=math.ceil(count/2); buffer=[];first=0;next_index=0
    try:
        for anchor in anchors:
            start=anchor//2
            while buffer and first<start: buffer.pop(0);first+=1
            last=min(start+32,total)
            while next_index<last:
                try: frame=next(stream)
                except StopIteration: raise ValueError('Source ended before declared video frame count')
                buffer.append(frame);next_index+=1
            clip=buffer[:last-start]
            if not clip: raise ValueError('Empty video window')
            clip=clip+[clip[-1]]*(32-len(clip))
            yield clip
    finally:
        stream.close()

def slowfast_model():
    from pytorchvideo.models.hub import slowfast_r50
    path=WORKSPACE/'work/model-cache/slowfast/SLOWFAST_8x8_R50.pyth'
    model=slowfast_r50(pretrained=False)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model_state'],strict=True)
    model.blocks[-1].proj=torch.nn.Identity();model.blocks[-1].activation=torch.nn.Identity()
    return model.eval().requires_grad_(False),path

def internvideo_model(checkpoint):
    package=WORKSPACE/'work/internvideo-upstream/InternVideo1/Pretrain/Multi-Modalities-Pretraining/InternVideo'
    for name,directory in [('_broader_iv',package),('_broader_iv.clip_utils',package/'clip_utils')]:
        namespace=types.ModuleType(name);namespace.__path__=[str(directory)];sys.modules[name]=namespace
    spec=importlib.util.spec_from_file_location('_broader_iv.clip_utils.model',package/'clip_utils/model.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)['state_dict']
    state={k.removeprefix('clip.'):v for k,v in state.items() if k.startswith('clip.')}
    if tuple(state['visual.conv1.weight'].shape)!=(1024,3,1,14,14): raise ValueError('Wrong InternVideo architecture')
    model=module.build_model(state)
    missing=[k for k in set(model.state_dict())-set(state) if k.startswith(('visual.','visual_ln_post.','visual_proj'))]
    if missing: raise ValueError(f'Incomplete visual checkpoint: {missing}')
    del model.transformer,model.token_embedding
    return model.float().eval().requires_grad_(False)

def text_features(cfg,model):
    import open_clip
    vocab=read(WORKSPACE/'work/chefost-data/MOSCATO/vocabulary/CMU/state_dict.json')
    states=sorted(s.strip().lower() for s in vocab['s2i'])
    objects=sorted({obj for split in ('train','validation','test') for _,obj in units(cfg,split)})
    encode=lambda prompts: torch.nn.functional.normalize(model.encode_text(open_clip.get_tokenizer(cfg['clip_model'])(prompts)).float(),dim=-1)
    with torch.inference_mode():
        embeddings={o:encode([f'A photo of {o} in the {s} state.' for s in states]).cpu() for o in objects}
        queries={o:encode([f'A photo of {o}.']).squeeze(0).cpu() for o in objects}
    torch.save({'states':states,'embeddings':embeddings,'queries':queries,'objects':objects},DATA/'text_embeddings.pt')

def onnx_session(backbone,half=False):
    import onnxruntime as ort
    root=WORKSPACE/'work/model-cache/onnx'
    name=backbone+('-half' if half else '')
    verified=read(root/(name+'-verification.json'))
    if verified['minimum_cosine']<.999:raise ValueError('ONNX encoder equivalence check failed')
    options=ort.SessionOptions();options.enable_mem_pattern=False
    options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    options.add_session_config_entry('session.inter_op.allow_spinning','0')
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session=ort.InferenceSession(str(root/(name+'.onnx')),sess_options=options,
                                providers=['DmlExecutionProvider','CPUExecutionProvider'])
    if 'DmlExecutionProvider' not in session.get_providers():raise RuntimeError('DirectML unavailable')
    return session,verified

def extract(cfg,backbone,video_id=None,max_windows=None,checkpoint=None,batch_size=8,backend='torch'):
    torch.set_num_threads(cfg['feature_threads'])
    smoke=max_windows is not None
    metadata_extra={'backbone':backbone,'labels_used_for_feature_extraction':False,'object_crops_used':False,
                    'sampling_seconds':cfg['sample_seconds'],'preprocessing':'short-side resize, center crop; backbone normalization',
                    'state': 'full-frame features; object-conditioned scorer has no tracker'}
    if backbone=='clip':
        import open_clip
        model,_,preprocess=open_clip.create_model_and_transforms(cfg['clip_model'],pretrained=cfg['clip_pretrained'])
        model.eval().requires_grad_(False)
        metadata_extra['model']=cfg['clip_model']+' / '+cfg['clip_pretrained']
        if not smoke and not (DATA/'text_embeddings.pt').exists(): text_features(cfg,model)
    elif backbone=='slowfast':
        model,path=slowfast_model()
        metadata_extra.update(model='official SlowFast R50 Kinetics400 8x8',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              input_frames=32,slow_frames=8,spatial_crop=256,window_raw_frames=64,normalization={'mean':.45,'std':.225})
    else:
        checkpoint=Path(checkpoint or WORKSPACE/'work/model-cache/internvideo/InternVideo-MM-L-14.ckpt')
        if not checkpoint.exists(): raise FileNotFoundError('Gated InternVideo checkpoint not available; supply --checkpoint')
        model=internvideo_model(checkpoint)
        metadata_extra.update(model='InternVideo-MM-L-14',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              input_frames=8,spatial_crop=224,window_raw_frames=64,
                              original_vidosc_extraction_parity_verified=False,object_features_zero_padded=True)
    session=None
    if backend in ('directml','directml_half'):
        if backbone=='internvideo':raise ValueError('InternVideo ONNX export not available')
        if backbone=='clip' and batch_size!=8:raise ValueError('CLIP ONNX export uses batches of eight')
        del model
        import gc;gc.collect()
        if backend=='directml_half' and backbone!='slowfast':raise ValueError('Half-precision verified only for SlowFast')
        session,verified=onnx_session(backbone,backend=='directml_half')
        metadata_extra.update(inference_backend='ONNX Runtime DirectML',numerical_verification=verified)
    else:metadata_extra['inference_backend']='PyTorch CPU'
    encoder=load_upstream_encoder() if backbone in ('clip','internvideo') else None
    videos=[video_id] if video_id else cfg['train']+cfg['validation']+cfg['test']
    for video in videos:
        target=feature_path(video,backbone,smoke);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            cached=torch.load(target,map_location='cpu',weights_only=True)
            if cached['metadata']['sampling_seconds']!=cfg['sample_seconds'] or cached['metadata']['backbone']!=backbone: raise ValueError('Feature cache mismatch')
            print('Cached',backbone,video,flush=True);continue
        path,meta=video_info(video)
        stride=int(round(meta['fps']*cfg['sample_seconds']))
        if stride%2: raise ValueError('Video window sampler requires even two-second raw-frame stride')
        anchors=list(range(0,meta['raw_frame_count'],stride))
        if smoke: anchors=anchors[:max_windows]
        meta.update(metadata_extra,sample_anchors=anchors,smoke_only=smoke)
        values=[];started=time.monotonic()
        print('Encoding',backbone,video,len(anchors),'samples',flush=True)
        with torch.inference_mode():
            if backbone=='clip':
                batch=[];stream=decoded_frames(path,224,stride)
                try:
                    for frame in stream:
                        batch.append(preprocess(Image.fromarray(frame)))
                        if len(batch)==batch_size or len(values)*batch_size+len(batch)==len(anchors):
                            if session is None:feat=model.encode_image(torch.stack(batch))
                            else:
                                padded=batch+[batch[-1]]*(8-len(batch))
                                feat=torch.from_numpy(session.run(None,{'images':torch.stack(padded).numpy()})[0][:len(batch)])
                            values.append(torch.nn.functional.normalize(feat.float(),dim=-1).cpu());batch=[]
                            print('Progress',video,sum(len(v) for v in values),'/',len(anchors),f'{time.monotonic()-started:.1f}s',flush=True)
                            if sum(len(v) for v in values)==len(anchors):break
                finally:stream.close()
            else:
                for i,frames in enumerate(video_windows(path,meta['raw_frame_count'],anchors)):
                    tensor=torch.from_numpy(np.stack(frames)).permute(3,0,1,2).float()/255
                    if backbone=='slowfast':
                        tensor=(tensor-.45)/.225
                        slow=tensor[:,torch.linspace(0,31,8).long()]
                        if session is None:feat=model([slow.unsqueeze(0),tensor.unsqueeze(0)])
                        else:feat=torch.from_numpy(session.run(None,{'slow':slow.unsqueeze(0).numpy(),'fast':tensor.unsqueeze(0).numpy()})[0])
                    else:
                        tensor=tensor[:,torch.linspace(0,31,8).long(),16:240,16:240]
                        mean=torch.tensor([.48145466,.4578275,.40821073])[:,None,None,None]
                        std=torch.tensor([.26862954,.26130258,.27577711])[:,None,None,None]
                        feat=model.encode_video(((tensor-mean)/std).unsqueeze(0))
                    values.append(torch.nn.functional.normalize(feat.float(),dim=-1).cpu())
                    if i==0 or (i+1)%10==0 or i+1==len(anchors): print('Progress',video,i+1,'/',len(anchors),f'{time.monotonic()-started:.1f}s',flush=True)
        visual=torch.cat(values)
        if len(visual)!=len(anchors) or not torch.isfinite(visual).all():raise ValueError('Invalid feature output')
        bundle={'visual':visual,'metadata':meta}
        if encoder is not None:
            with torch.inference_mode():bundle['temporal']=encoder(torch.cat([visual,torch.zeros_like(visual)],dim=-1).unsqueeze(0),middle=True).squeeze(0).cpu()
        meta['extraction_seconds']=time.monotonic()-started
        partial=target.with_suffix('.partial');torch.save(bundle,partial);partial.replace(target)
        write(target.with_suffix('.json'),meta)
        print('Saved',target,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['clip','slowfast','internvideo'],required=True)
    p.add_argument('--video-id');p.add_argument('--max-windows',type=int);p.add_argument('--checkpoint');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--backend',choices=['torch','directml','directml_half'],default='torch')
    p.add_argument('--protocol',default=str(CONFIG))
    args=p.parse_args();extract(read(args.protocol),args.backbone,args.video_id,args.max_windows,args.checkpoint,args.batch_size,args.backend)
