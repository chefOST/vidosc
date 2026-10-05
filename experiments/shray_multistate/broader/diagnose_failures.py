"""Post-evaluation diagnostics; does not modify models or predictions."""
import sys
import torch,numpy as np
import protocol
sys.path.insert(0,str(protocol.HERE))
from protocol import HERE,DATA,WORKSPACE,read,write
from metrics import annotation_matrix
from temporal_states import smooth_scores,expand_samples

def main():
    torch.set_num_threads(8);comparison=read(HERE/'results/comparison.json')
    name=comparison['selection_lock']['selected_regular_model'];directory=HERE/'results'/name
    checkpoint=torch.load(directory/'model.pt',weights_only=True);head=checkpoint['head']
    text=torch.load(DATA/'text_embeddings.pt',weights_only=True);priors=read(directory/'selection.json')['priors']
    diagnostics={}
    for obj,query in text['queries'].items():
        z=(query-head['feature_mean'][1280:2048])/head['feature_std'][1280:2048]
        offset=text['embeddings'][obj].mean(dim=0)-head['text_center']
        diagnostics[obj]={'object_query_seen_in_training':obj in priors,
            'absolute_normalized_query_max':float(z.abs().max()),
            'absolute_normalized_query_p95':float(torch.quantile(z.abs(),.95)),
            'candidate_mean_offset_from_fixed_training_center_norm':float(offset.norm())}
    saved=torch.load(directory/'test-predictions.pt',weights_only=False)
    key='S16_Eggs_7150991-858::egg mixture';meta=saved['metadata'][key.split('::')[0]]
    ann=read(DATA/'test_labels.json')[key]
    vocab=read(WORKSPACE/'work/chefost-data/MOSCATO/vocabulary/CMU/state_dict.json')
    aliases={k.strip().lower():v.strip().lower() for k,v in vocab.get('all2one',{}).items()}
    truth,valid=annotation_matrix(meta,ann,saved['states'],aliases)
    dense=expand_samples(smooth_scores(saved['methods']['trained'][key],saved['policies']['trained']['smoothing']),
                         meta['sample_anchors'],meta['raw_frame_count'])
    top=dense[valid].argmax(axis=1);labels,counts=np.unique(top,return_counts=True)
    unit=comparison['results'][name]['methods']['trained']['units'][key]
    write(HERE/'failure-analysis.json',{'selected_model':name,'object_query_normalization':diagnostics,
       'interpretation':'Large normalized unseen-query inputs and changed candidate noun offsets are possible contributors to unstable unseen-object scores; these diagnostics do not establish a causal explanation.',
       'egg_mixture_example':{'top1_counts_on_known_frames':{saved['states'][i]:int(c) for i,c in zip(labels,counts)},
           'top1_hit':unit['top1_frame_hit_accuracy'],'multi_label_f1':unit['micro_f1'],
           'exact_label_set_accuracy':unit['exact_state_set_accuracy'],'gt_states':unit['gt_unique_states'],
           'predicted_states':unit['predicted_unique_states_on_annotated_frames']},
       'next_experiment':'Test query-based candidate centering, normalization for unseen object names, loss treatment for unseen state candidates, and real object features; use new held-out recordings for a fresh accuracy claim.',
       'models_or_predictions_modified':False})
    print('Saved post-evaluation failure diagnostics',flush=True)

if __name__=='__main__':main()
