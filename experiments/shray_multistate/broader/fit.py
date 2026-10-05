"""Train broader object-conditioned state models without opening test labels."""
import argparse,copy,hashlib,time
from pathlib import Path
import numpy as np
import torch
from protocol import CONFIG,DATA,HERE,WORKSPACE,read,write,assert_disjoint
from features import feature_path
from model import QueryStateModel
from metrics import ValidationCounts,choose_policy
from scoring import sampled_targets

def source_backbone(mode):return 'internvideo' if mode=='internvideo' else 'clip'

def load_bundles(cfg,split,mode):
    bundles={};hashes={}
    for video in cfg[split]:
        path=feature_path(video,source_backbone(mode))
        bundle=torch.load(path,map_location='cpu',weights_only=True)
        if bundle['metadata'].get('smoke_only'):raise ValueError('Cannot train/evaluate on smoke features')
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if mode=='clip_slowfast':
            motion_path=feature_path(video,'slowfast')
            motion=torch.load(motion_path,map_location='cpu',weights_only=True)
            for k in ('source_sha256','sample_anchors','raw_frame_count','fps'):
                if bundle['metadata'][k]!=motion['metadata'][k]:raise ValueError('Motion/visual alignment mismatch')
            bundle['motion']=motion['visual'];hashes[str(motion_path)]=hashlib.sha256(motion_path.read_bytes()).hexdigest()
        bundles[video]=bundle
    return bundles,hashes

def fit(mode,heldout=False,epochs=None,finetune_epochs=None,protocol_path=CONFIG):
    protocol_path=Path(protocol_path);cfg=read(protocol_path);assert_disjoint(cfg)
    name=mode+cfg.get('run_suffix','')+('-heldout-peeled' if heldout else '')
    out=HERE/'results'/name
    if out.exists():raise FileExistsError(f'Use a new protocol/run directory: {out}')
    train_label_file=DATA/cfg.get('train_labels_file','train_labels.json')
    train_annotations=read(train_label_file);val_annotations=read(DATA/'validation_labels.json')
    train_bundles,feature_hashes=load_bundles(cfg,'train',mode)
    val_bundles,val_hashes=load_bundles(cfg,'validation',mode);feature_hashes.update(val_hashes)
    text=torch.load(DATA/'text_embeddings.pt',map_location='cpu',weights_only=True)
    states=text['states'];heldout_names=cfg['heldout_states'] if heldout else []
    columns=[i for i,s in enumerate(states) if s not in heldout_names]
    vocab=read(WORKSPACE/'work/chefost-data/MOSCATO/vocabulary/CMU/state_dict.json')
    aliases={k.strip().lower():v.strip().lower() for k,v in vocab.get('all2one',{}).items()}
    torch.set_num_threads(cfg['feature_threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    model=QueryStateModel(2304 if mode=='clip_slowfast' else 0)
    train_units={};prior_rows={}
    for key,annotation in train_annotations.items():
        bundle=train_bundles[annotation['video_id']];obj=annotation['object']
        source_annotation=annotation
        if annotation['fps']==bundle['metadata']['fps'] and len(annotation['states'])<bundle['metadata']['raw_frame_count']:
            annotation={**annotation,'states':annotation['states']+[[]]*(bundle['metadata']['raw_frame_count']-len(annotation['states']))}
        targets,mask=sampled_targets(bundle,annotation,states,aliases)
        if heldout_names:mask &= ~targets[:,[states.index(s) for s in heldout_names]].bool().any(dim=1)
        if not mask.any():continue
        x=model.features(bundle['visual'],text['queries'][obj],bundle.get('motion'),bundle['temporal']).detach()[mask]
        train_units[key]={'x':x,'y':targets[mask][:,columns],'full_y':targets[mask],'mask':mask,'bundle':bundle,
                          'query':text['queries'][obj],'texts':text['embeddings'][obj][columns]}
        prior_rows.setdefault(obj,[]).append(targets[mask].mean(dim=0).numpy())
    if not train_units:raise ValueError('No training examples')
    y=torch.cat([u['y'] for u in train_units.values()]);x=torch.cat([u['x'] for u in train_units.values()])
    positive=y.sum(dim=0);weights=((len(y)-positive)/positive.clamp_min(1)).sqrt().clamp(1,4);weights[positive==0]=1
    model.head.feature_mean.copy_(x.mean(dim=0));model.head.feature_std.copy_(x.std(dim=0).clamp_min(.01))
    train_objects=sorted(prior_rows)
    model.head.text_center.copy_(torch.cat([text['embeddings'][o][columns] for o in train_objects]).mean(dim=0))
    val_units={};counts={}
    for key,annotation in val_annotations.items():
        bundle=val_bundles[annotation['video_id']];obj=annotation['object']
        val_units[key]={'bundle':bundle,'query':text['queries'][obj],'texts':text['embeddings'][obj][columns]}
        counts[key]=ValidationCounts(bundle['metadata'],annotation,states,aliases,columns,heldout_names)
    def validation(use_cached):
        model.eval()
        with torch.inference_mode():
            contexts={video:b['temporal'] if use_cached else model.context(b['visual']) for video,b in val_bundles.items()}
            scores={k:model(u['bundle']['visual'],u['query'],u['texts'],u['bundle'].get('motion'),
                            contexts[u['bundle']['metadata']['video_id']]).sigmoid().numpy() for k,u in val_units.items()}
        return choose_policy(scores,counts)
    out.mkdir(parents=True)
    manifest={'mode':mode,'config':cfg,'protocol_file':str(protocol_path.resolve()),'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'test_labels_loaded_for_training':False,'test_features_loaded_for_training':False,
              'train_label_sha256':hashlib.sha256(train_label_file.read_bytes()).hexdigest(),
              'validation_label_sha256':hashlib.sha256((DATA/'validation_labels.json').read_bytes()).hexdigest(),
              'text_sha256':hashlib.sha256((DATA/'text_embeddings.pt').read_bytes()).hexdigest(),
              'feature_hashes':feature_hashes,'training_object_video_units':len(train_units),'training_labelled_samples':len(y),
              'training_candidate_states':[states[i] for i in columns],'heldout_states':heldout_names,
              'training_states_with_positive_samples':[states[columns[i]] for i in range(len(columns)) if positive[i]>0],
              'training_candidate_states_with_no_positive_samples':[states[columns[i]] for i in range(len(columns)) if positive[i]==0],
              'heldout_states_absent_from_training_candidates':all(s not in [states[i] for i in columns] for s in heldout_names),
              'loss':'equal object/video-weighted BCE; sqrt positive weights capped at 4',
              'query_conditioning':'CLIP object-name embedding concatenated with visual and VidOSC context; no object tracker',
              'architecture_input_dim':x.shape[1],'image_text_and_motion_encoders_frozen':True}
    write(out/'training_manifest.json',manifest)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-3,weight_decay=1e-3)
    best=None;history=[];started=time.monotonic()
    for epoch in range(1,(epochs or cfg['epochs'])+1):
        model.train();optimizer.zero_grad();loss_value=0
        for unit in train_units.values():
            loss=loss_fn(model.head(unit['x'],unit['texts']),unit['y'])/len(train_units)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite head loss')
            loss.backward();loss_value+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.head.parameters(),5);optimizer.step()
        if epoch==1 or epoch%20==0 or epoch==(epochs or cfg['epochs']):
            policy=validation(True);history.append({'stage':'head','epoch':epoch,'loss':loss_value,'validation':policy})
            if best is None or (policy['micro_f1'],policy['precision'])>(best['policy']['micro_f1'],best['policy']['precision']):
                best={'head':copy.deepcopy(model.head.state_dict()),'encoder':copy.deepcopy(model.encoder.state_dict()),'policy':policy,'stage':'head','epoch':epoch}
            print(name,'head',epoch,'loss',round(loss_value,4),'valF1',round(policy['micro_f1'],4),'seconds',round(time.monotonic()-started),flush=True)
    frozen=copy.deepcopy(best)
    model.head.load_state_dict(best['head']);model.encoder.load_state_dict(best['encoder'])
    model.train_encoder=True;model.encoder.requires_grad_(True)
    optimizer=torch.optim.AdamW([{'params':model.head.parameters(),'lr':2e-4},
        {'params':list(model.encoder.proj.parameters())+list(model.encoder.ln.parameters()),'lr':1e-5},
        {'params':model.encoder.transformer_encoder.parameters(),'lr':1e-6}],weight_decay=1e-3)
    grouped={}
    for unit in train_units.values():grouped.setdefault(unit['bundle']['metadata']['video_id'],[]).append(unit)
    for epoch in range(1,(finetune_epochs if finetune_epochs is not None else cfg['finetune_epochs'])+1):
        model.train();optimizer.zero_grad();loss_value=0
        for group in grouped.values():
            b=group[0]['bundle'];context=model.context(b['visual']);group_loss=0
            for unit in group:
                logits=model(b['visual'],unit['query'],unit['texts'],b.get('motion'),context)[unit['mask']]
                group_loss=group_loss+loss_fn(logits,unit['y'])/len(train_units)
            if not torch.isfinite(group_loss):raise FloatingPointError('Nonfinite temporal loss')
            group_loss.backward();loss_value+=float(group_loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if epoch==1 or epoch%10==0 or epoch==(finetune_epochs if finetune_epochs is not None else cfg['finetune_epochs']):
            policy=validation(False);history.append({'stage':'finetune','epoch':epoch,'loss':loss_value,'validation':policy})
            if (policy['micro_f1'],policy['precision'])>(best['policy']['micro_f1'],best['policy']['precision']):
                best={'head':copy.deepcopy(model.head.state_dict()),'encoder':copy.deepcopy(model.encoder.state_dict()),'policy':policy,'stage':'finetune','epoch':epoch}
            print(name,'finetune',epoch,'loss',round(loss_value,4),'valF1',round(policy['micro_f1'],4),'seconds',round(time.monotonic()-started),flush=True)
    checkpoint={**best,'mode':mode,'motion_dim':model.motion_dim,'heldout_states':heldout_names,'config':cfg,
                'protocol_file':manifest['protocol_file'],
                'protocol_sha256':manifest['protocol_sha256'],'text_sha256':manifest['text_sha256'],'test_labels_loaded_for_training':False}
    torch.save(checkpoint,out/'model.pt')
    torch.save({**checkpoint,**frozen},out/'frozen-model.pt')
    write(out/'training_history.json',history)
    priors={obj:np.stack(rows).mean(axis=0) for obj,rows in prior_rows.items()}
    fallback=np.stack(list(priors.values())).mean(axis=0)
    prior_scores={k:np.tile(priors.get(u['object'],fallback)[columns],(len(val_bundles[u['video_id']]['visual']),1)) for k,u in val_annotations.items()}
    clip_bundles,_=load_bundles(cfg,'validation','clip')
    clip_scores={k:torch.sigmoid(10*(clip_bundles[u['video_id']]['visual']@text['embeddings'][u['object']][columns].T)-2).numpy() for k,u in val_annotations.items()}
    baseline_policies={'prior':choose_policy(prior_scores,counts),'clip':choose_policy(clip_scores,counts)}
    write(out/'selection.json',{'selected_model':{'stage':best['stage'],'epoch':best['epoch'],'policy':best['policy']},
                              'frozen_model':{'epoch':frozen['epoch'],'policy':frozen['policy']},
                              'baselines':baseline_policies,'priors':{o:v.tolist() for o,v in priors.items()},'fallback_prior':fallback.tolist(),
                              'selection_used_test_labels':False})
    print('Finished',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['clip','clip_slowfast','internvideo'],required=True)
    p.add_argument('--heldout',action='store_true');p.add_argument('--epochs',type=int);p.add_argument('--finetune-epochs',type=int)
    p.add_argument('--protocol',default=str(CONFIG))
    args=p.parse_args();fit(args.mode,args.heldout,args.epochs,args.finetune_epochs,args.protocol)
