"""Finite validation-selected accuracy experiment; fresh-test prediction firewall."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'broader'))
import protocol
from protocol import DATA,WORKSPACE,read,write,assert_disjoint
import numpy as np
import torch
from model import QueryStateModel
from models import StableQueryModel,normalized_visual,prototype_scores
from features import feature_path
from metrics import ValidationCounts,choose_policy,evaluate,aggregate
from temporal_states import smooth_scores

CFG=HERE/'protocol.json'
OUT=HERE/'results'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bundles(cfg,split):
    result={}
    for video in cfg[split]:
        path=feature_path(video,'clip')
        if not path.exists():raise FileNotFoundError(f'Full CLIP features needed: {path}')
        result[video]=torch.load(path,map_location='cpu',weights_only=True)
        assert not result[video]['metadata'].get('smoke_only')
    return result

def prepare_units(annotations,features,states,aliases,text):
    result={}
    for key,ann in annotations.items():
        b=features[ann['video_id']]
        counts=ValidationCounts(b['metadata'],ann,states,aliases)
        assert counts.known.sum()>0,key
        result[key]={'bundle':b,'object':ann['object'],'query':text['queries'][ann['object']],
                     'candidates':text['embeddings'][ann['object']], 'counts':counts,
                     'targets':torch.from_numpy((counts.positives/np.maximum(counts.known,1)).astype('float32')),
                     'weights':torch.from_numpy(counts.known[:,0].astype('float32'))}
    return result

def neural_scores(model,units):
    model.eval()
    with torch.inference_mode():
        return {k:model(u['bundle']['visual'],u['query'],u['candidates'],
                        temporal=u['bundle']['temporal']).sigmoid().numpy() for k,u in units.items()}

def fit_neural(mode,train,val,cfg):
    torch.manual_seed(cfg['seed'])
    model=(StableQueryModel() if mode=='stable' else QueryStateModel()).eval()
    with torch.no_grad():
        for unit in train.values():
            unit['features']=model.features(unit['bundle']['visual'],unit['query'],temporal=unit['bundle']['temporal'])
        all_features=torch.cat([u['features'][u['weights']>0] for u in train.values()])
        if mode=='stable':model.initialize_statistics(all_features)
        else:
            model.head.feature_mean.copy_(all_features.mean(0))
            model.head.feature_std.copy_(all_features.std(0).clamp_min(.01))
        objects=sorted({u['object'] for u in train.values()})
        model.head.text_center.copy_(torch.cat([next(u['candidates'] for u in train.values() if u['object']==o) for o in objects]).mean(0))
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-3,weight_decay=1e-3)
    criterion=torch.nn.BCEWithLogitsLoss(reduction='none')
    best=None;history=[];counts={k:u['counts'] for k,u in val.items()}
    for epoch in range(1,cfg['head_epochs']+1):
        model.train();optimizer.zero_grad();total=0
        for u in train.values():
            if mode=='stable':
                x=(u['features']-model.head.feature_mean)/model.head.feature_std
                visual=torch.nn.functional.normalize(model.head.projection(x),dim=-1)
                texts=torch.nn.functional.normalize(u['candidates']-u['query'],dim=-1)
                logits=visual@texts.T*model.head.log_scale.clamp(0,np.log(100)).exp()+model.head.bias
            else:logits=model.head(u['features'],u['candidates'])
            loss=(criterion(logits,u['targets']).mean(1)*u['weights']).sum()/u['weights'].sum()/len(train)
            assert torch.isfinite(loss)
            loss.backward();total+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.head.parameters(),5);optimizer.step()
        if epoch==1 or epoch%10==0:
            scores=neural_scores(model,val);policy=choose_policy(scores,counts)
            history.append({'epoch':epoch,'loss':total,'validation':policy})
            if best is None or (policy['micro_f1'],policy['precision'])>(best['policy']['micro_f1'],best['policy']['precision']):
                best={'head':copy.deepcopy(model.head.state_dict()),'encoder':copy.deepcopy(model.encoder.state_dict()),
                      'mode':mode,'epoch':epoch,'policy':policy}
            print(mode,epoch,'val F1',round(policy['micro_f1'],4),flush=True)
    model.head.load_state_dict(best['head'])
    torch.save(best,OUT/(mode+'.pt'));write(OUT/(mode+'-history.json'),history)
    return model,best

def fit_prototypes(train,states):
    raw=torch.cat([torch.cat([u['bundle']['visual'],u['bundle']['temporal']],-1)[u['weights']>0] for u in train.values()])
    mean=raw.mean(0);std=raw.std(0).clamp_min(.01)
    def entry(rows):
        pos=neg=positive_count=total_count=None
        for u in rows:
            z=normalized_visual(u['bundle'],mean,std);weights=u['weights'][:,None];y=u['targets']
            p=(y*weights).T@z;n=((1-y)*weights).T@z
            count=(y*weights).sum(0);total=u['weights'].sum()
            if pos is None:pos=p;neg=n;positive_count=count;total_count=total
            else:pos+=p;neg+=n;positive_count+=count;total_count+=total
        # Unit prototype direction, not an output neuron for each supplied text.
        return {'positive':torch.nn.functional.normalize(pos,dim=-1),
                'negative':torch.nn.functional.normalize(neg,dim=-1),
                'prior':positive_count/total_count}
    fitted={'states':states,'mean':mean,'std':std,'global':entry(list(train.values())),
            'by_object':{o:entry([u for u in train.values() if u['object']==o]) for o in sorted({u['object'] for u in train.values()})}}
    torch.save(fitted,OUT/'prototypes.pt')
    return fitted

def select_blend(scores,priors,counts,weights):
    best=None
    for weight in weights:
        mixed={k:weight*scores[k]+(1-weight)*priors[k] for k in scores}
        policy=choose_policy(mixed,counts)
        candidate={'neural_weight':weight,'policy':policy}
        if best is None or (policy['micro_f1'],policy['precision'])>(best['policy']['micro_f1'],best['policy']['precision']):best=candidate
    return best

def main():
    cfg=read(CFG);assert_disjoint(cfg);torch.set_num_threads(cfg['feature_threads'])
    if OUT.exists():raise FileExistsError('Do not overwrite a selected/evaluated run')
    train_b=bundles(cfg,'train');val_b=bundles(cfg,'validation')
    text=torch.load(DATA/'text_embeddings.pt',map_location='cpu',weights_only=True)
    vocab=read(WORKSPACE/'work/chefost-data/MOSCATO/vocabulary/CMU/state_dict.json')
    aliases={k.strip().lower():v.strip().lower() for k,v in vocab.get('all2one',{}).items()}
    train=prepare_units(read(DATA/'v2_train_labels.json'),train_b,text['states'],aliases,text)
    val=prepare_units(read(DATA/'v2_validation_labels.json'),val_b,text['states'],aliases,text)
    OUT.mkdir();models={};selections={};counts={k:u['counts'] for k,u in val.items()}
    fitted=fit_prototypes(train,text['states'])
    prior_scores={k:prototype_scores(u['bundle'],u['object'],fitted,0) for k,u in val.items()}
    selections['prior']={'policy':choose_policy(prior_scores,counts)}
    for mode in cfg['neural_modes']:
        model,checkpoint=fit_neural(mode,train,val,cfg);models[mode]=model
        scores=neural_scores(model,val);selections[mode]={'policy':checkpoint['policy'],'epoch':checkpoint['epoch']}
        selections[mode+'_blend']=select_blend(scores,prior_scores,counts,cfg['prior_mixture_weights'])
    for scale in cfg['prototype_scales']:
        scores={k:prototype_scores(u['bundle'],u['object'],fitted,scale) for k,u in val.items()}
        policy=choose_policy(scores,counts)
        if 'prototype' not in selections or (policy['micro_f1'],policy['precision'])>(selections['prototype']['policy']['micro_f1'],selections['prototype']['policy']['precision']):
            selections['prototype']={'scale':scale,'policy':policy}
    legacy_path=WORKSPACE/'outputs/shray-multistate/broader/results/clip-manual/model.pt'
    legacy=torch.load(legacy_path,map_location='cpu',weights_only=True)
    old=QueryStateModel().eval();old.head.load_state_dict(legacy['head']);old.encoder.load_state_dict(legacy['encoder'])
    selections['legacy']={'policy':legacy['policy'],'checkpoint_sha256':sha(legacy_path),
                          'note':'unchanged original selected model; original threshold/smoothing'}
    eligible=[k for k in selections if k!='legacy']
    winner=max(eligible,key=lambda k:(selections[k]['policy']['micro_f1'],selections[k]['policy']['precision']))
    lock={'protocol_sha256':sha(CFG),'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},
          'training_labels_sha256':sha(DATA/'v2_train_labels.json'),'validation_labels_sha256':sha(DATA/'v2_validation_labels.json'),
          'models_sha256':{p.name:sha(p) for p in OUT.glob('*.pt')},'selections':selections,'winner':winner,
          'fresh_test_labels_loaded':False,'fresh_test_scores_used':False}
    write(OUT/'selection-lock.json',lock)
    print('Locked validation winner:',winner,'F1',selections[winner]['policy']['micro_f1'],flush=True)
    deadline=time.monotonic()+3600
    while not all(feature_path(v,'clip').exists() for v in cfg['test']):
        if time.monotonic()>deadline:raise TimeoutError('Fresh test features not ready; selection lock remains preserved')
        time.sleep(5)
    test_b=bundles(cfg,'test');pred={name:{} for name in selections}
    with torch.inference_mode():
        for video in cfg['test']:
            b=test_b[video]
            for obj in cfg['targets'][video]:
                key=video+'::'+obj;prior=prototype_scores(b,obj,fitted,0);pred['prior'][key]=prior
                pred['prototype'][key]=prototype_scores(b,obj,fitted,selections['prototype']['scale'])
                for mode,model in models.items():
                    score=model(b['visual'],text['queries'][obj],text['embeddings'][obj],temporal=b['temporal']).sigmoid().numpy()
                    pred[mode][key]=score
                    w=selections[mode+'_blend']['neural_weight'];pred[mode+'_blend'][key]=w*score+(1-w)*prior
                # Legacy context must be computed by its fine-tuned encoder.
                pred['legacy'][key]=old(b['visual'],text['queries'][obj],text['embeddings'][obj]).sigmoid().numpy()
    prediction_path=OUT/'fresh-test-predictions.pt'
    torch.save({'states':text['states'],'predictions':pred,'metadata':{v:b['metadata'] for v,b in test_b.items()},'selection':lock},prediction_path)
    write(OUT/'prediction-lock.json',{'sha256':sha(prediction_path),'all_methods_saved_before_test_labels':True,
                                    'feature_hashes':{v:sha(feature_path(v,'clip')) for v in cfg['test']}})
    # First fresh-test annotation export/scoring is after all predictions lock.
    gt=read(WORKSPACE/'work/chefost-data/MOSCATO/annotations/ground_truth/CMU/gt_annotations_cmu.json')
    annotations={v+'::'+o:{'video_id':v,'object':o,'fps':30,'states':gt[v]['state'][o]}
                 for v in cfg['test'] for o in cfg['targets'][v]}
    write(DATA/'v2_fresh_test_labels.json',annotations)
    results={}
    for name,by_unit in pred.items():
        rows={k:evaluate(scores,test_b[annotations[k]['video_id']]['metadata'],annotations[k],text['states'],aliases,selections[name]['policy']) for k,scores in by_unit.items()}
        results[name]={'aggregate':aggregate(rows),'units':rows}
        print(name,'FRESH F1',round(results[name]['aggregate']['macro_over_object_video_units']['micro_f1'],4),flush=True)
    write(OUT/'evaluation.json',results)
    for path,digest in read(HERE/'prior-artifact-hashes.json').items():assert sha(WORKSPACE/path)==digest,path
    report(cfg,lock,results)

def report(cfg,lock,results):
    winner=lock['winner'];metric=lambda name:results[name]['aggregate']['macro_over_object_video_units']
    rows=['| Method | Fresh F1 | Highest-label hit | Exact label set |','|---|---:|---:|---:|']
    for name in results:
        m=metric(name);rows.append(f"| {name} | {100*m['micro_f1']:.1f}% | {100*m['top1_frame_hit_accuracy']:.1f}% | {100*m['exact_state_set_accuracy']:.1f}% |")
    summary={'validation_selected_model':winner,'fresh_test_f1':metric(winner)['micro_f1'],
             'prior_f1':metric('prior')['micro_f1'],'unchanged_legacy_f1':metric('legacy')['micro_f1'],
             'test_recordings':len(cfg['test']),'test_units':len(results[winner]['units']),
             'beats_same_data_prior':metric(winner)['micro_f1']>metric('prior')['micro_f1'],
             'original_artifacts_preserved':True,'scope':cfg['scope']}
    write(OUT/'summary.json',summary)
    body=f"""# Fresh accuracy experiment

Validation selected **{winner}** before fresh test scoring. Its mean target/video
multi-label F1 is **{100*summary['fresh_test_f1']:.1f}%**, compared with
**{100*summary['prior_f1']:.1f}%** for the same-training-data prior and
**{100*summary['unchanged_legacy_f1']:.1f}%** for the unchanged original checkpoint
on these same new recordings. Original results and checkpoints are preserved.

"""+'\n'.join(rows)+f"""

Nine human-labelled training recordings and five validation recordings were
used. Old exposed tests are development data now. The fresh test has four new
subjects/recordings, six targets, in the same CMU kitchen. This evaluates known
target categories, including pizza examples now present in training; it does
not establish arbitrary-object or unseen-state recognition. Split planning
used video/object names and source annotations for splitting, not fresh scores.

The original-head and stable models use the same new data/loss/budget. Stable
uses fixed query scaling and query-relative text centering. Prototypes learn
positive/negative visual-temporal directions for supported object/state names;
this calibration does not establish recognition of unseen descriptions.
All models use frozen CLIP/VidOSC features. There are no object crops/tracker.

Every method's prediction was saved and hashed before fresh annotation export
and scoring. Selection locks and per-unit/interval metrics are in results/.
Scores are means over target/video units; frames repeated from two-second
samples are correlated. The earlier 30.3% result was on a different test set.
Use the unchanged legacy model scored here for a direct comparison.

Michelle and Shray pursue similar semantic-state goals. Her pipeline obtains
descriptions with Claude after VidOSC phases; this experiment trains/calibrates
temporal state scores. Architectural difference alone is not proof of novelty
or better recognition.
"""
    (HERE/'REPORT.md').write_text(body,encoding='utf-8')
    deliver=WORKSPACE/'outputs/shray-accuracy-v2';deliver.mkdir(parents=True,exist_ok=True)
    (deliver/'REPORT.md').write_text(body,encoding='utf-8');write(deliver/'summary.json',summary)

if __name__=='__main__':main()
