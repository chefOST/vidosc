"""Lock validation choices, save test predictions, THEN open test annotations."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import protocol
sys.path.insert(0,str(protocol.HERE))
from protocol import CONFIG,DATA,HERE,WORKSPACE,read,write,units
from fit import load_bundles
from model import QueryStateModel
from metrics import evaluate,aggregate
from temporal_states import smooth_scores,state_intervals

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def lock_selection(names):
    rows={}
    for name in names:
        directory=HERE/'results'/name
        selected=read(directory/'selection.json')['selected_model']
        rows[name]={'validation':selected,'checkpoint_sha256':sha(directory/'model.pt'),
                    'selection_sha256':sha(directory/'selection.json'),
                    'training_protocol_sha256':read(directory/'training_manifest.json')['protocol_sha256']}
    ordinary=[n for n in names if 'heldout' not in n]
    winner=max(ordinary,key=lambda n:(rows[n]['validation']['policy']['micro_f1'],rows[n]['validation']['policy']['precision']))
    lock={'protocol_sha256':sha(CONFIG),'runs':rows,'selected_regular_model':winner,
          'selection_criterion':'validation-only equal object/video mean micro-F1; precision breaks ties',
          'test_labels_used':False,'test_scores_used':False}
    path=HERE/'results/comparison-selection-lock.json'
    if path.exists() and read(path)!=lock:raise ValueError('Model selection lock already exists and differs')
    write(path,lock)
    return lock

def predict_run(name):
    out=HERE/'results'/name;cfg=read(CONFIG)
    checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    protocol_file=Path(checkpoint.get('protocol_file',CONFIG))
    if checkpoint['protocol_sha256']!=sha(protocol_file):raise ValueError('Training protocol changed')
    for split in ('validation','test'):
        if checkpoint['config'][split]!=cfg[split]:raise ValueError('Compared methods must share validation/test videos')
    if checkpoint['text_sha256']!=sha(DATA/'text_embeddings.pt'):raise ValueError('Text embeddings changed')
    text=torch.load(DATA/'text_embeddings.pt',map_location='cpu',weights_only=True)
    bundles,hashes=load_bundles(cfg,'test',checkpoint['mode'])
    clips,_=load_bundles(cfg,'test','clip')
    selection=read(out/'selection.json');states=text['states']
    model=QueryStateModel(checkpoint['motion_dim']).eval()
    model.head.load_state_dict(checkpoint['head']);model.encoder.load_state_dict(checkpoint['encoder'])
    frozen=torch.load(out/'frozen-model.pt',map_location='cpu',weights_only=True)
    methods={m:{} for m in ('trained','frozen','prior','raw_clip')}
    policies={'trained':checkpoint['policy'],'frozen':frozen['policy'],
              'prior':selection['baselines']['prior'],'raw_clip':selection['baselines']['clip']}
    with torch.inference_mode():
        for video,obj in units(cfg,'test'):
            b=bundles[video];key=video+'::'+obj
            methods['trained'][key]=model(b['visual'],text['queries'][obj],text['embeddings'][obj],b.get('motion')).sigmoid().numpy()
            prior=np.asarray(selection['priors'].get(obj,selection['fallback_prior']))
            methods['prior'][key]=np.tile(prior,(len(b['visual']),1))
            methods['raw_clip'][key]=torch.sigmoid(10*(clips[video]['visual']@text['embeddings'][obj].T)-2).numpy()
        model.head.load_state_dict(frozen['head']);model.encoder.load_state_dict(frozen['encoder'])
        for video,obj in units(cfg,'test'):
            b=bundles[video];key=video+'::'+obj
            methods['frozen'][key]=model(b['visual'],text['queries'][obj],text['embeddings'][obj],b.get('motion')).sigmoid().numpy()
    payload={'states':states,'methods':methods,'policies':policies,'metadata':{v:b['metadata'] for v,b in bundles.items()},
             'heldout_states':checkpoint['heldout_states'],'test_labels_opened':False,
             'test_feature_hashes':hashes,'checkpoint_sha256':sha(out/'model.pt'),'protocol_sha256':sha(protocol_file)}
    path=out/'test-predictions.pt'
    if path.exists():raise FileExistsError('Preserve existing test predictions; do not rerun/cherry-pick')
    torch.save(payload,path)
    # A manifest commits outputs before any ground-truth reads in this process.
    write(out/'prediction-manifest.json',{'prediction_sha256':sha(path),'test_labels_opened_before_prediction_save':False,
          'checkpoint_sha256':payload['checkpoint_sha256'],'protocol_sha256':sha(CONFIG),'test_feature_hashes':hashes,
          'candidate_states':states,'object_video_units':len(units(cfg,'test'))})
    timelines={}
    for key,scores in methods['trained'].items():
        meta=bundles[key.split('::')[0]]['metadata'];policy=policies['trained']
        timelines[key]=state_intervals(smooth_scores(scores,policy['smoothing'])>=policy['threshold'],states,
                                      meta['sample_anchors'],meta['raw_frame_count'],meta['fps'])
    write(out/'predicted-timelines.json',timelines)
    print('Saved predictions before reading test labels:',name,flush=True)

def score_run(name):
    out=HERE/'results'/name;path=out/'test-predictions.pt';manifest=read(out/'prediction-manifest.json')
    if manifest['prediction_sha256']!=sha(path):raise ValueError('Saved prediction hash mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    annotations=read(DATA/'test_labels.json')
    vocab=read(WORKSPACE/'work/chefost-data/MOSCATO/vocabulary/CMU/state_dict.json')
    aliases={k.strip().lower():v.strip().lower() for k,v in vocab.get('all2one',{}).items()}
    result={'run':name,'protocol_sha256':sha(CONFIG),'prediction_sha256':sha(path),
            'test_label_sha256':sha(DATA/'test_labels.json'),'candidate_states':saved['states'],
            'definitions':{'primary':'mean object/video micro-F1, with every object/video equally weighted',
               'top1':'fraction of annotated raw frames whose highest-score state belongs to the human label set',
               'macro_state_f1':'mean F1 over GT-present states within each unit; extra states penalize micro-F1',
               'segments':'same-label maximum-cardinality one-to-one interval matching; unknown frames split intervals',
               'timing':'mean endpoint error for matched segments at IoU 0.5 only; inspect with segment recall',
               'sampling':'one prediction every two seconds expanded onto raw frames; frames are correlated',
               'state_count':'number of distinct labels predicted on known annotation frames; candidate inventory is fixed at 77'},
            'methods':{}}
    for method,by_unit in saved['methods'].items():
        rows={key:evaluate(scores,saved['metadata'][annotations[key]['video_id']],annotations[key],saved['states'],aliases,
                           saved['policies'][method],saved['heldout_states']) for key,scores in by_unit.items()}
        seen={k:r for k,r in rows.items() if '_Pizza_' not in k};pizza={k:r for k,r in rows.items() if '_Pizza_' in k}
        result['methods'][method]={'policy':saved['policies'][method],'all':aggregate(rows),
                                   'seen_recipes':aggregate(seen),'unseen_pizza_recipe_and_objects':aggregate(pizza),'units':rows}
    write(out/'evaluation.json',result)
    print(name,'TEST mean unit F1',round(result['methods']['trained']['all']['macro_over_object_video_units']['micro_f1'],4),flush=True)
    return result

def compare(names):
    torch.set_num_threads(read(CONFIG)['feature_threads'])
    lock=lock_selection(names)
    # Complete predictions for every run before opening the test ground truth.
    for name in names:predict_run(name)
    results={name:score_run(name) for name in names}
    write(HERE/'results/comparison.json',{'selection_lock':lock,'results':results,
          'internvideo_comparison':'pending checkpoint' if 'internvideo' not in names else 'completed'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    compare(p.parse_args().runs)
