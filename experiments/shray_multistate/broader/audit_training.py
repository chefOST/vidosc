"""Training-only pseudo-label coverage and visual inspection sheet."""
import collections,subprocess,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from protocol import DATA,HERE,read,write
from features import FFMPEG

def audit():
    annotations=read(DATA/'train_labels.json');rows={};selected=[]
    for key,ann in annotations.items():
        sequence=ann['states'];counts=collections.Counter(s for labels in sequence for s in set(labels))
        rows[key]={'total_frames_at_10fps':len(sequence),'empty_frames':sum(not labels for labels in sequence),
                   'multi_label_frames':sum(len(set(labels))>1 for labels in sequence),'state_positive_frame_counts':dict(counts),
                   'label_source':'automatic MOSCATO pseudo labels, not manual human ground truth'}
        if ann['video_id'].startswith('S07_') and ann['object']!='egg mixture':
            known=[i for i,labels in enumerate(sequence) if labels]
            for fraction in (.25,.5,.75):
                i=known[min(int(len(known)*fraction),len(known)-1)]
                selected.append((ann,i))
    root=HERE/'training-audit';root.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(3*420,4*420),'#f5f5f5');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for j,(ann,index) in enumerate(selected):
        path=root/(ann['video_id']+f'-{index}.png');timestamp=index/ann['fps']
        if not path.exists():subprocess.run([str(FFMPEG),'-v','error','-ss',str(timestamp),'-i',str(DATA/(ann['video_id']+'.avi')),
                    '-frames:v','1','-vf','scale=400:-2',str(path)],check=True)
        frame=Image.open(path).convert('RGB');x=(j%3)*420+10;y=(j//3)*420+10
        sheet.paste(frame,(x,y));label=f"{ann['video_id'].split('_')[1]} / {ann['object']} / {timestamp:.1f}s\nPseudo label: {', '.join(sorted(set(ann['states'][index])))}"
        for line_no,line in enumerate(textwrap.wrap(label.replace('\n',' | '),49)):
            draw.text((x,y+frame.height+5+line_no*19),line,font=font,fill='#171717')
    sheet.save(root/'training-pseudo-label-samples.png')
    write(root/'coverage.json',{'units':rows,'inspection_sheet':'training-pseudo-label-samples.png',
                              'sampling':'25/50/75 percent positions among nonempty training pseudo-label frames',
                              'human_annotation_corrections_applied':False})
    print('Training-only audit saved',root,flush=True)

if __name__=='__main__':audit()
