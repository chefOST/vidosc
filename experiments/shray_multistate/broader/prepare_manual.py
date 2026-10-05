"""Validation-driven manual-label comparison, declared before any fresh test scoring."""
import copy,hashlib
from datetime import datetime,timezone
from protocol import CONFIG,DATA,HERE,WORKSPACE,read,write,assert_disjoint

MANUAL=HERE/'protocol-manual.json'
def main():
    if (HERE/'results/comparison-selection-lock.json').exists():raise RuntimeError('Do not add methods after test selection is locked')
    if list((HERE/'results').glob('*/test-predictions.pt')):raise RuntimeError('Do not add methods after test predictions exist')
    cfg=copy.deepcopy(read(CONFIG))
    cfg.update(train=['S41_Salad_7150991-349','S08_Eggs_7150991-727','S24_Brownie_7150991-483','S12_Sandwich_7150991-2470'],
               training_label_fps=30,training='four human-ground-truth training videos; subjects separate from validation/test',
               run_suffix='-manual',train_labels_file='manual_train_labels.json',
               scope='Manual-supervised comparison with four training videos and four target units; same locked validation and fresh test recordings as the 12-video pseudo-supervised pilot. S08 has no separate egg-mixture training annotation.',
               feature_backbones=['clip'],comparison_models=['clip-manual'])
    assert_disjoint(cfg)
    if MANUAL.exists() and read(MANUAL)!=cfg:raise RuntimeError('Manual protocol changed')
    write(MANUAL,cfg)
    targets={'S41_Salad_7150991-349':'vegetables','S08_Eggs_7150991-727':'eggs',
             'S24_Brownie_7150991-483':'eggs','S12_Sandwich_7150991-2470':'bread slices'}
    gt=read(WORKSPACE/'work/chefost-data/MOSCATO/annotations/ground_truth/CMU/gt_annotations_cmu.json')
    annotations={v+'::'+o:{'video_id':v,'object':o,'fps':30,'states':gt[v]['state'][o]} for v,o in targets.items()}
    write(DATA/'manual_train_labels.json',annotations)
    write(HERE/'manual-comparison-plan.json',{'declared_utc':datetime.now(timezone.utc).isoformat(),
       'trigger':'Pseudo-supervised CLIP validation F1 approximately ties constant prior; training audit shows persistent automatic labels.',
       'choice':'Compare four-human-video training using identical architecture, epoch budgets, validation/test units, and selection metric.',
       'test_labels_or_scores_used_for_this_choice':False,'test_prediction_files_exist':False,
       'source':'Published human CMU annotations, copied only into the own workspace; source files unchanged.',
       'manual_train_videos':cfg['train'],'manual_train_units':4,'egg_mixture_direct_training_available':False,
       'confounds':'Both training recordings and supervision source change; this does not isolate a causal label-quality effect.',
       'protocol_sha256':hashlib.sha256(MANUAL.read_bytes()).hexdigest(),
       'training_labels_sha256':hashlib.sha256((DATA/'manual_train_labels.json').read_bytes()).hexdigest()})
    print('Declared manual-supervised comparison before test prediction/scoring',flush=True)

if __name__=='__main__':main()
