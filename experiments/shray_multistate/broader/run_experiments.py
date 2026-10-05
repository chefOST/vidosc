"""Finite local runner: wait for feature jobs, fit models, then evaluate once."""
import subprocess,time,sys,hashlib
import protocol
sys.path.insert(0,str(protocol.HERE))
from protocol import CONFIG,DATA,HERE,WORKSPACE,read,write
from features import feature_path

def wait_features(backbones,splits,config_path=CONFIG):
    cfg=read(config_path);paths=[feature_path(v,b) for s in splits for v in cfg[s] for b in backbones]
    while not all(p.exists() for p in paths):
        done=sum(p.exists() for p in paths)
        write(HERE/'progress.json',{'stage':'waiting for frozen features','backbones':backbones,'splits':splits,
                                  'completed_files':done,'expected_files':len(paths),'test_accuracy_available':False})
        print('Waiting for',backbones,splits,done,'/',len(paths),flush=True)
        time.sleep(30)

def run(arguments):
    subprocess.run([sys.executable,*map(str,arguments)],cwd=WORKSPACE,check=True)

def main():
    for mode,heldout,config_path in [('clip',False,CONFIG),('clip',True,CONFIG),
                                    ('clip',False,HERE/'protocol-manual.json'),('clip_slowfast',False,CONFIG)]:
        needed=['clip','slowfast'] if mode=='clip_slowfast' else ['clip']
        wait_features(needed,['train','validation'],config_path)
        name=mode+read(config_path).get('run_suffix','')+('-heldout-peeled' if heldout else '')
        out=HERE/'results'/name
        if (out/'model.pt').exists() and (out/'selection.json').exists():continue
        if out.exists():raise RuntimeError(f'Incomplete training directory requires inspection: {out}')
        write(HERE/'progress.json',{'stage':'training','run':name,'test_accuracy_available':False})
        run([HERE/'fit.py','--mode',mode,'--protocol',config_path,*(['--heldout'] if heldout else [])])
    wait_features(['clip','slowfast'],['test'])
    runs=['clip','clip_slowfast','clip-manual','clip-heldout-peeled']
    write(HERE/'progress.json',{'stage':'test prediction and evaluation','test_accuracy_available':False})
    run([HERE/'check_checkpoints.py'])
    run([HERE/'evaluate_broader.py','--runs',*runs])
    run([HERE/'report_broader.py'])
    write(HERE/'progress.json',{'stage':'completed CLIP, motion fusion, and held-out-state experiments',
                              'test_accuracy_available':True,'internvideo':'pending gated pretrained checkpoint'})

if __name__=='__main__':main()
