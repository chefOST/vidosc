"""Verify real trained checkpoints with 3/5/77 candidates on a training recording."""
import sys,hashlib
import numpy as np
import torch
import protocol
sys.path.insert(0,str(protocol.HERE))
from protocol import DATA,HERE,WORKSPACE,write,read
from features import feature_path
from model import QueryStateModel
from prepare import load_upstream_encoder

def main():
    torch.set_num_threads(8)
    text=torch.load(DATA/'text_embeddings.pt',weights_only=True)
    original=load_upstream_encoder();results={}
    for name in ('clip','clip_slowfast','clip-manual','clip-heldout-peeled'):
        directory=HERE/'results'/name;checkpoint=torch.load(directory/'model.pt',weights_only=True)
        manifest=read(directory/'training_manifest.json');video=checkpoint['config']['train'][0]
        obj='vegetables';bundle=torch.load(feature_path(video,'clip'),weights_only=True)
        motion=torch.load(feature_path(video,'slowfast'),weights_only=True)['visual'] if checkpoint['motion_dim'] else None
        model=QueryStateModel(checkpoint['motion_dim']).eval()
        model.encoder.load_state_dict(checkpoint['encoder']);model.head.load_state_dict(checkpoint['head'])
        with torch.inference_mode():
            context=model.context(bundle['visual'])
            full=model(bundle['visual'],text['queries'][obj],text['embeddings'][obj],motion,context)
            errors={}
            for count in (3,5,77):
                indices=np.random.default_rng(42).permutation(77)[:count]
                subset=model(bundle['visual'],text['queries'][obj],text['embeddings'][obj][indices],motion,context)
                torch.testing.assert_close(subset,full[:,indices],rtol=1e-4,atol=1e-4)
                errors[str(count)]=(subset-full[:,indices]).abs().max().item()
            alternate=model(bundle['visual'],text['queries']['eggs'],text['embeddings'][obj],motion,context)
        query_change=(alternate-full).abs().max().item()
        assert query_change>.001 and torch.isfinite(full).all()
        changed=sum(not torch.equal(v,original.state_dict()[k]) for k,v in checkpoint['encoder'].items())
        if checkpoint['stage']=='head':assert changed==0
        else:assert changed>0
        for state in checkpoint['heldout_states']:
            assert state not in manifest['training_candidate_states']
            assert state not in manifest['training_states_with_positive_samples']
        results[name]={'training_video_used':video,'test_labels_opened':False,'candidate_subset_max_logit_error':errors,
           'object_query_max_logit_change':query_change,'selected_stage':checkpoint['stage'],
           'encoder_state_tensors_changed_from_original':changed,'output_shape':list(full.shape),'all_scores_finite':True,
           'heldout_labels_excluded_from_training':True,'checkpoint_sha256':hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest()}
        print('Checkpoint interface verified',name,flush=True)
    write(HERE/'checkpoint-interface-verification.json',{'runs':results,
          'upstream_model_source_sha256':hashlib.sha256((WORKSPACE/'work/VidOSC/model.py').read_bytes()).hexdigest()})

if __name__=='__main__':main()
