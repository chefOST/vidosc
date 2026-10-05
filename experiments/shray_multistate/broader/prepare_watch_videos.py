"""Playable copies of fresh evaluation recordings; no annotation reads."""
import subprocess,json
from protocol import HERE,DATA,CONFIG,read,write
from features import FFMPEG,FFPROBE,video_info

def main():
    root=HERE/'test-videos';root.mkdir(exist_ok=True);rows={}
    for video in read(CONFIG)['test']:
        source,metadata=video_info(video);target=root/(video+'.mp4')
        if not target.exists():subprocess.run([str(FFMPEG),'-v','error','-i',str(source),'-map','0:v:0',
                      '-c','copy','-movflags','+faststart',str(target)],check=True)
        probe=json.loads(subprocess.check_output([str(FFPROBE),'-v','error','-select_streams','v:0','-count_frames',
                  '-show_entries','stream=nb_read_frames,avg_frame_rate','-of','json',str(target)]))['streams'][0]
        assert int(probe['nb_read_frames'])==metadata['raw_frame_count']
        preview=root/(video+'.png')
        if not preview.exists():subprocess.run([str(FFMPEG),'-v','error','-ss',str(metadata['raw_frame_count']/metadata['fps']/2),
                 '-i',str(target),'-frames:v','1',str(preview)],check=True)
        rows[video]={'path':str(target),'raw_frame_count':metadata['raw_frame_count'],
                     'duration_seconds':metadata['raw_frame_count']/metadata['fps'],'video_stream_copied_without_reencoding':True}
        print('Playable and frame-count verified',video,flush=True)
    write(root/'manifest.json',rows)
    (root/'README.md').write_text('# Fresh test recordings\n\nThese five videos are the new subject-disjoint evaluation recordings. '
       'None is used for training or validation in the locked broader protocol. They show different recipes and people but remain in the same CMU kitchen. '
       'MP4 files copy the original video stream without reencoding; complete decoded frame counts were verified.\n',encoding='utf-8')

if __name__=='__main__':main()
