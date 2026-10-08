"""Declare a fresh-test protocol and prepare training/validation labels only."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'broader'))
from protocol import WORKSPACE, DATA, read, write, assert_disjoint

TRAIN = ['S41_Salad_7150991-349','S08_Eggs_7150991-727','S24_Brownie_7150991-483',
         'S12_Sandwich_7150991-2470','S15_Salad_7150991-451','S16_Eggs_7150991-858',
         'S19_Brownie_7150991-848','S36_Sandwich_7150991-165','S15_Pizza_7150991-5341']
VAL = ['S29_Salad_7150991-1','S09_Eggs_7150991-1050','S13_Brownie_7150991-631',
       'S31_Sandwich_7150991-1','S20_Pizza_7150991-180']
TEST = ['S17_Salad_7150991-822','S14_Eggs_7150991-1016',
        'S37_Brownie_7150991-133','S55_Pizza_7150991-292']

def main():
    # Names/object inventories plan the split; no fresh-test labels are exported
    # or scored before model/prediction locking. Former test recordings are now
    # explicitly development data, never presented as a fresh test again.
    gt = read(WORKSPACE/'work/chefost-data/MOSCATO/annotations/ground_truth/CMU/gt_annotations_cmu.json')
    desired={'Salad':['vegetables'],'Eggs':['eggs','egg mixture'],
             'Brownie':['eggs','egg mixture'],'Sandwich':['bread slices'],'Pizza':['cheese','batter']}
    targets={v:[o for o in desired[v.split('_')[1]] if o in gt[v]['state']] for v in TRAIN+VAL+TEST}
    cfg={'seed':42,'train':TRAIN,'validation':VAL,'test':TEST,'targets':targets,
         'sample_seconds':2,'feature_threads':8,'clip_model':'ViT-L-14-quickgelu','clip_pretrained':'openai',
         'objects_by_recipe':desired,'training_label_fps':30,'evaluation_label_fps':30,
         'primary_metric':'equal target/video mean micro-F1 on known annotation frames',
         'training_source':'human annotations; old exposed tests reassigned to development',
         'scope':'new subjects/recordings, known recipes and target categories; same kitchen; not unseen-object recognition',
         'split_planning':'video and object names inspected; full annotation source parsed only for splitting; no new-test scores inspected',
         'head_epochs':100,'neural_modes':['original','stable'],
         'stable_changes':['fixed query scaling','query-relative text centering'],
         'common_changes':['more human labels','raw-frame-weighted soft bin targets','unweighted BCE'],
         'prototype_scales':[0,4,12,32], 'prior_mixture_weights':[0,.25,.5,.75,1],
         'selection':'validation-only mean F1 then precision; mixture weight zero is a prior-only control, not a learned success',
         'prediction_lock':'all methods saved before fresh test annotation export/scoring'}
    assert_disjoint(cfg)
    if (HERE/'protocol.json').exists():
        assert read(HERE/'protocol.json')==cfg, 'Preserve the declared protocol'
    else: write(HERE/'protocol.json',cfg)
    for split in ('train','validation'):
        labels={v+'::'+o:{'video_id':v,'object':o,'fps':30,'states':gt[v]['state'][o]}
                for v in cfg[split] for o in targets[v]}
        write(DATA/f'v2_{split}_labels.json',labels)
    snapshots={str(p.relative_to(WORKSPACE)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (WORKSPACE/'outputs/shray-multistate/broader/results').rglob('*')
               if p.is_file() and p.suffix in {'.pt','.json'}}
    write(HERE/'prior-artifact-hashes.json',snapshots)
    print(json.dumps({'train_videos':len(TRAIN),'validation_videos':len(VAL),'fresh_test_videos':len(TEST),
                      'targets':{s:sum(len(targets[v]) for v in cfg[s]) for s in ('train','validation','test')}}),flush=True)

if __name__=='__main__':main()
