"""Locked, subject-disjoint broader pilot; annotation preparation is separate from training."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare import WORKSPACE, read, write

HERE=Path(__file__).resolve().parent
DATA=WORKSPACE/'work/shray-broader'
CONFIG=HERE/'protocol.json'

def create():
    train_ids=['S07_Salad_7150991-2095','S07_Eggs_7150991-1466','S07_Brownie_7150991-1431','S07_Sandwich_7150991-2343',
               'S18_Salad_7150991-649','S18_Eggs_7150991-3391','S18_Brownie_7150991-885','S18_Sandwich_7150991-893',
               'S33_Salad_7150991-85','S33_Eggs_7150991-461','S33_Brownie_7150991-211','S33_Sandwich_7150991-3']
    val_ids=['S29_Salad_7150991-1','S09_Eggs_7150991-1050','S13_Brownie_7150991-631','S31_Sandwich_7150991-1']
    test_ids=['S15_Salad_7150991-451','S16_Eggs_7150991-858','S19_Brownie_7150991-848','S36_Sandwich_7150991-165','S20_Pizza_7150991-180']
    objects={'Salad':['vegetables'],'Eggs':['eggs','egg mixture'],'Brownie':['eggs'],
             'Sandwich':['bread slices'],'Pizza':['cheese','batter']}
    cfg={'seed':42,'sample_seconds':2,'training_label_fps':10,'evaluation_label_fps':30,
         'clip_model':'ViT-L-14-quickgelu','clip_pretrained':'openai','heldout_states':['peeled'],
         'train':train_ids,'validation':val_ids,'test':test_ids,'objects_by_recipe':objects,
         'primary_metric':'macro mean object/video micro-F1; all methods use validation-only global threshold/smoothing',
         'training':'CMU MOSCATO pseudo-labels only; all manual GT video IDs excluded',
         'scope':'Custom subject-disjoint 12-training/4-validation/5-test CMU pilot across four training recipes, with pizza recipe/object transfer test. Same CMU kitchen; cross-kitchen generalization is not measured.',
         'previously_examined_development':['S29_Salad_7150991-1','S41_Salad_7150991-349','S48_Salad_7150991-222'],
         'test_labels_examined_for_model_selection':False,
         'split_planning':'Video/object names and label inventories were inspected to plan coverage. This is not a blinded or official full benchmark.',
         'feature_backbones':['clip','slowfast','internvideo'],'comparison_models':['clip','clip_slowfast','internvideo'],
         'feature_threads':8,'epochs':200,'finetune_epochs':40,
         'heldout_state_protocol':'Remove held-out names from training candidates/loss and ignore supervised training/validation frames carrying those names. Temporal context and pretrained encoders may still contain visual exposure; this tests unseen labelled states, not never-seen pixels.'}
    assert_disjoint(cfg)
    if CONFIG.exists():
        if read(CONFIG)!=cfg: raise ValueError('Existing locked protocol differs')
    else: write(CONFIG,cfg)
    return cfg

def assert_disjoint(cfg):
    groups=[set(cfg[x]) for x in ('train','validation','test')]
    subjects=[{v.split('_')[0] for v in g} for g in groups]
    for a,b in ((0,1),(0,2),(1,2)):
        if groups[a]&groups[b] or subjects[a]&subjects[b]: raise ValueError('Video or subject leakage')

def units(cfg,split):
    return [(v,obj) for v in cfg[split] for obj in cfg['objects_by_recipe'][v.split('_')[1]]]

def labels(cfg):
    annotation_root=WORKSPACE/'work/chefost-data/MOSCATO/annotations'
    gt=read(annotation_root/'ground_truth/CMU/gt_annotations_cmu.json')
    pseudo=read(annotation_root/'pseudo_labels/CMU/pseudo_annotations_poc_new_cmu.json')
    if set(cfg['train'])&set(gt): raise ValueError('Training cannot use manual benchmark videos')
    manifest={'protocol_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),'splits':{}}
    for split in ('train','validation','test'):
        selected={}
        for video,obj in units(cfg,split):
            source=pseudo if split=='train' else gt
            key=video+'::'+obj
            selected[key]={'video_id':video,'object':obj,'fps':cfg['training_label_fps'] if split=='train' else cfg['evaluation_label_fps'],
                           'states':source[video]['state'][obj]}
        path=DATA/f'{split}_labels.json'
        write(path,selected)
        manifest['splits'][split]={'videos':cfg[split],'object_video_units':len(selected),'label_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    write(DATA/'label_manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':
    labels(create())
