"""Preflight only train/validation annotation clocks; test checked after predictions."""
import argparse
from pathlib import Path
from protocol import CONFIG,DATA,HERE,WORKSPACE,read,write
from features import video_info
from scoring import label_matrix

def main(protocol_path=CONFIG):
    cfg=read(protocol_path);vocab=read(WORKSPACE/'work/chefost-data/MOSCATO/vocabulary/CMU/state_dict.json')
    states=sorted(s.strip().lower() for s in vocab['s2i'])
    aliases={k.strip().lower():v.strip().lower() for k,v in vocab.get('all2one',{}).items()};metadata={};rows={}
    for split in ('train','validation'):
        label_file=cfg.get('train_labels_file','train_labels.json') if split=='train' else 'validation_labels.json'
        for key,ann in read(DATA/label_file).items():
            video=ann['video_id']
            if video not in metadata:metadata[video]=video_info(video)[1]
            meta=metadata[video]
            difference=abs(len(ann['states'])/ann['fps']-meta['raw_frame_count']/meta['fps'])
            if ann['fps']==meta['fps']:assert len(ann['states'])<=meta['raw_frame_count'],key
            else:assert difference<=.2,(key,difference)
            target,valid=label_matrix(ann['states'],states,aliases)
            rows[key]={'split':split,'video_fps':meta['fps'],'annotation_fps':ann['fps'],'raw_video_frames':meta['raw_frame_count'],
                       'annotation_frames':len(ann['states']),'duration_difference_seconds':difference,
                       'trailing_frames_outside_annotation_coverage':max(0,meta['raw_frame_count']-len(ann['states'])) if ann['fps']==meta['fps'] else 0,
                       'all_labels_canonical_or_known_alias':True,'known_annotation_frames':int(valid.sum())}
    target='annotation-alignment-validation'+cfg.get('run_suffix','')+'.json'
    write(HERE/target,{'passed':True,'test_labels_opened':False,'units':rows})
    print('All',len(rows),'train/validation object/video clocks and label vocabularies passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',default=str(CONFIG));main(p.parse_args().protocol)
