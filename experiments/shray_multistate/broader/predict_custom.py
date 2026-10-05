"""Run an own checkpoint with any candidate count and optional state descriptions."""
import argparse,sys
from pathlib import Path
import torch
import protocol
sys.path.insert(0,str(protocol.HERE))
from protocol import DATA,HERE,read,write
from features import feature_path
from model import QueryStateModel
from temporal_states import smooth_scores,state_intervals

def predict(checkpoint_path,video,obj,candidate_path,output):
    torch.set_num_threads(8)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    backbone='internvideo' if checkpoint['mode']=='internvideo' else 'clip'
    bundle=torch.load(feature_path(video,backbone),map_location='cpu',weights_only=True)
    if bundle['metadata'].get('smoke_only'):raise ValueError('Full video features required')
    candidates=read(candidate_path)
    if not isinstance(candidates,list) or not candidates:raise ValueError('Candidates must be a nonempty list')
    items=[{'state':c,'description':None} if isinstance(c,str) else c for c in candidates]
    labels=[c['state'] for c in items]
    if len(set(labels))!=len(labels):raise ValueError('Candidate state names must be unique')
    text=torch.load(DATA/'text_embeddings.pt',map_location='cpu',weights_only=True)
    cached=obj in text['objects'] and all(c['state'] in text['states'] and not c.get('description') for c in items)
    if cached:
        query=text['queries'][obj];embeddings=text['embeddings'][obj][[text['states'].index(s) for s in labels]]
    else:
        import open_clip
        model,_,_=open_clip.create_model_and_transforms(checkpoint['config']['clip_model'],pretrained='openai')
        model.eval();tokenizer=open_clip.get_tokenizer(checkpoint['config']['clip_model'])
        prompts=[c.get('description') or f"A photo of {obj} in the {c['state']} state." for c in items]
        with torch.inference_mode():
            embeddings=torch.nn.functional.normalize(model.encode_text(tokenizer(prompts)).float(),dim=-1)
            query=torch.nn.functional.normalize(model.encode_text(tokenizer([f'A photo of {obj}.'])).float(),dim=-1)[0]
        del model
    motion=None
    if checkpoint['motion_dim']:
        motion_bundle=torch.load(feature_path(video,'slowfast'),map_location='cpu',weights_only=True)
        for k in ('source_sha256','sample_anchors','raw_frame_count','fps'):
            if bundle['metadata'][k]!=motion_bundle['metadata'][k]:raise ValueError('Motion/visual features misaligned')
        motion=motion_bundle['visual']
    model=QueryStateModel(checkpoint['motion_dim']).eval()
    model.head.load_state_dict(checkpoint['head']);model.encoder.load_state_dict(checkpoint['encoder'])
    with torch.inference_mode():scores=model(bundle['visual'],query,embeddings,motion).sigmoid().numpy()
    policy=checkpoint['policy'];smoothed=smooth_scores(scores,policy['smoothing']);meta=bundle['metadata']
    active=smoothed>=policy['threshold']
    result={'video_id':video,'object':obj,'candidate_states':items,'number_of_candidates':len(items),
            'predicted_unique_states':[s for i,s in enumerate(labels) if active[:,i].any()],
            'intervals':state_intervals(active,labels,meta['sample_anchors'],meta['raw_frame_count'],meta['fps']),
            'sample_anchors':meta['sample_anchors'],'scores':smoothed.tolist(),'policy':policy,
            'score_meaning':'sigmoid confidence scores, not calibrated probabilities; simultaneous states allowed',
            'scope':'Custom descriptions and new object names are supported as inputs; accuracy for them is not established by the CMU evaluation.'}
    target=Path(output).resolve()
    if not target.is_relative_to(HERE):raise ValueError('Save user results inside the own broader module')
    write(target,result);print('Saved',target,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--video-id',required=True)
    p.add_argument('--object',required=True);p.add_argument('--candidates',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();predict(a.checkpoint,a.video_id,a.object,a.candidates,a.output)
