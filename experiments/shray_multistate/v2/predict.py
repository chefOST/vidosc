"""Inference from the validation-selected v2 model with a supplied candidate list."""
import argparse
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'broader'))
import protocol
from protocol import DATA,read,write
import torch
from model import QueryStateModel
from models import StableQueryModel,prototype_scores
from features import feature_path
from temporal_states import smooth_scores,state_intervals

def predict(video,obj,candidate_file,output):
    torch.set_num_threads(8)
    lock=read(HERE/'results/selection-lock.json');selected=lock['winner']
    items=read(candidate_file)
    if not isinstance(items,list) or not items:raise ValueError('Supply a nonempty candidate list')
    items=[{'state':i} if isinstance(i,str) else i for i in items]
    names=[i['state'] for i in items]
    if len(set(names))!=len(names):raise ValueError('Candidate names must be unique')
    text=torch.load(DATA/'text_embeddings.pt',map_location='cpu',weights_only=True)
    cached=obj in text['queries'] and all(i['state'] in text['states'] and not i.get('description') for i in items)
    if cached:
        query=text['queries'][obj];candidates=text['embeddings'][obj][[text['states'].index(n) for n in names]]
    else:
        import open_clip
        clip,_,_=open_clip.create_model_and_transforms('ViT-L-14-quickgelu',pretrained='openai');clip.eval()
        tokenizer=open_clip.get_tokenizer('ViT-L-14-quickgelu')
        with torch.inference_mode():
            query=torch.nn.functional.normalize(clip.encode_text(tokenizer([f'A photo of {obj}.'])).float(),dim=-1)[0]
            candidates=torch.nn.functional.normalize(clip.encode_text(tokenizer([i.get('description') or f"A photo of {obj} in the {i['state']} state." for i in items])).float(),dim=-1)
        del clip
    bundle=torch.load(feature_path(video,'clip'),map_location='cpu',weights_only=True)
    mode='original' if selected.startswith('original') else 'stable'
    checkpoint=torch.load(HERE/'results'/f'{mode}.pt',map_location='cpu',weights_only=True)
    model=(StableQueryModel() if mode=='stable' else QueryStateModel()).eval()
    model.head.load_state_dict(checkpoint['head']);model.encoder.load_state_dict(checkpoint['encoder'])
    with torch.inference_mode():
        scores=model(bundle['visual'],query,candidates,temporal=bundle['temporal']).sigmoid().numpy()
    if selected in ('prototype','prior') or selected.endswith('_blend'):
        fit=torch.load(HERE/'results/prototypes.pt',map_location='cpu',weights_only=True)
        scale=lock['selections']['prototype']['scale'] if selected=='prototype' else 0
        calibrated=prototype_scores(bundle,obj,fit,scale,names,scores)
        if selected.endswith('_blend'):
            weight=lock['selections'][selected]['neural_weight'];scores=weight*scores+(1-weight)*calibrated
        else:scores=calibrated
    policy=lock['selections'][selected]['policy'];scores=smooth_scores(scores,policy['smoothing'])
    meta=bundle['metadata']
    result={'selected_method':selected,'video_id':video,'object':obj,'candidates':items,
            'intervals':state_intervals(scores>=policy['threshold'],names,meta['sample_anchors'],meta['raw_frame_count'],meta['fps']),
            'scores':scores.tolist(),'sample_anchors':meta['sample_anchors'],'policy':policy,
            'scope':'supplied names/descriptions; arbitrary new descriptions/objects are not validated; scores are not calibrated probabilities'}
    target=Path(output).resolve()
    if not target.is_relative_to(HERE/'local-predictions'):raise ValueError('Save inference under v2/local-predictions')
    write(target,result);print('Saved',target,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video-id',required=True);p.add_argument('--object',required=True)
    p.add_argument('--candidates',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();predict(args.video_id,args.object,args.candidates,args.output)
