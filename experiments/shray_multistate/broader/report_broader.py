"""Plain-language report and scientific comparison plot from saved evaluations."""
import sys
import numpy as np
import protocol
sys.path.insert(0,str(protocol.HERE))
from protocol import HERE,DATA,read,write,CONFIG

def pct(x):return f'{100*x:.1f}%'

def main():
    comparison=read(HERE/'results/comparison.json');lock=comparison['selection_lock'];results=comparison['results']
    best=lock['selected_regular_model'];summary=[]
    labels=[];f1s=[];top1s=[]
    for name in ('clip','clip_slowfast','clip-manual'):
        for method in ('trained','frozen','prior','raw_clip'):
            item=results[name]['methods'][method];m=item['all']['macro_over_object_video_units']
            summary.append(f"| {name} / {method} | {pct(m['micro_f1'])} | {pct(m['top1_frame_hit_accuracy'])} | {pct(m['macro_f1_gt_present'])} | {pct(item['all']['segment_f1_macro_over_units']['0.5'])} | {m['unique_state_count_absolute_error']:.2f} |")
            if name=='clip' or method=='trained':
                labels.append(name+' / '+method);f1s.append(m['micro_f1']);top1s.append(m['top1_frame_hit_accuracy'])
    selected=results[best]['methods']['trained'];m=selected['all']['macro_over_object_video_units']
    baseline=results[best]['methods']['prior']['all']['macro_over_object_video_units']
    # Paired resampling by whole recording, because expanded frames are correlated.
    differences={k:r['micro_f1']-results[best]['methods']['prior']['units'][k]['micro_f1'] for k,r in selected['units'].items()}
    videos=sorted({k.split('::')[0] for k in differences});rng=np.random.default_rng(42);resampled=[]
    for _ in range(5000):
        sampled=rng.choice(videos,len(videos),replace=True)
        resampled.append(np.mean([value for v in sampled for k,value in differences.items() if k.split('::')[0]==v]))
    interval=np.quantile(resampled,[.025,.975]).tolist()
    write(HERE/'bootstrap-f1-lift.json',{'comparison':'validation-selected model minus matched prior',
          'unit':'paired whole-video resampling, keeping all targets within each sampled recording','seed':42,
          'replicates':5000,'recordings':len(videos),'mean_unit_f1_difference':float(np.mean(list(differences.values()))),
          'exploratory_95_percentile_interval':interval,'warning':'Five recordings from one kitchen; not a broad reliability guarantee.'})
    units=[]
    for key,r in selected['units'].items():
        units.append(f"| {key} | {pct(r['micro_f1'])} | {pct(r['top1_frame_hit_accuracy'])} | {len(r['gt_unique_states'])} | {len(r['predicted_unique_states_on_annotated_frames'])} | {', '.join(r['missed_state_names']) or 'none'} |")
    held=results['clip-heldout-peeled']['methods']['trained'];heldrows=[r['heldout_states']['peeled'] for r in held['units'].values() if r['heldout_states']['peeled']['gt_positive_frames']]
    heldf1=float(np.mean([r['f1'] for r in heldrows])) if heldrows else None
    seen=selected['seen_recipes']['macro_over_object_video_units'];pizza=selected['unseen_pizza_recipe_and_objects']['macro_over_object_video_units']
    verdict='beats' if m['micro_f1']>baseline['micro_f1'] else 'does not beat'
    strongest_prior=max(results[n]['methods']['prior']['all']['macro_over_object_video_units']['micro_f1'] for n in ('clip','clip_slowfast','clip-manual'))
    lines=["# Broader multi-state VidOSC results",'',
        f"The model selected using validation alone was **{best}**. On the seven fresh object/video test units, it achieved **{pct(m['micro_f1'])} mean multi-label F1** and **{pct(m['top1_frame_hit_accuracy'])} mean top-one label-hit accuracy**. It {verdict} the matched constant-prior baseline's {pct(baseline['micro_f1'])} mean F1.",'',
        f"**The accuracy goal is not achieved.** All learned variants score below the stronger pseudo-training constant prior's **{pct(strongest_prior)} F1**. The selected model reaches only {pct(pizza['micro_f1'])} F1 on the unseen pizza targets, and held-out peeled detection scores zero F1. The implementation supports variable candidates, but reliable unseen-object or unseen-state recognition has not been demonstrated.",'',
        "The primary custom CMU pilot uses 21 videos: 12 pseudo-labelled training videos, four human-labelled validation videos, and five fresh human-labelled test videos. A separate manual-supervised comparison adds four human-labelled training recordings, for 25 unique videos overall. Subjects are separate within each comparison. The seven test units include two targets in eggs and two in pizza. All videos are still in the CMU kitchen; this is not evidence about other kitchens or arbitrary foods.",'',
        f"An exploratory paired bootstrap over whole test recordings gives a 95% interval for the F1 difference versus prior of **{100*interval[0]:+.1f} to {100*interval[1]:+.1f} percentage points**. There are only five recordings, so treat this as an indication of uncertainty rather than a broad reliability guarantee.",'',
        "## What the percentages mean",'',
        "Top-one label-hit accuracy asks whether the single highest-score state belongs to the human label set at an annotated time. It can look good even while the model predicts too many extra states. Multi-label F1 measures the balance between finding correct states and avoiding extra states, including simultaneous states. The table's percentages are means across object/video units, not one pooled percentage over all frames. A two-second sample is repeated across raw frames; those frames are not independent trials.",'',
        "These scores measure agreement with the dataset's named labels. Each target is an ingredient/object category such as vegetables or eggs, potentially combining several physical instances. This evaluation does not measure per-instance tracking, object presence, or segmentation accuracy.",'',
        "Segment F1 at IoU 0.5 tests whether a predicted named interval overlaps a human interval enough, with one-to-one matching. State-count error is the average absolute difference between the number of distinct predicted and human-labelled states on annotated frames. Exact label-set accuracy and per-state precision/recall/F1 are in the evaluation JSON files.",'',
        "## Comparisons",'',
        '| Model / method | Mean F1 | Mean top-one hit | Mean state F1 | Segment F1 @ .5 | Count error |',
        '|---|---:|---:|---:|---:|---:|',*summary,'',
        "All epochs, thresholds, smoothing, and the regular-model winner were chosen using validation, then locked before test evaluation. 'Frozen' fits the own text head while retaining the original VidOSC encoder; 'trained' additionally considers fine-tuning, retained only if validation improves. Prior uses per-object training label frequencies, with a global fallback for new objects. Raw CLIP compares image and text embeddings without fitting VidOSC. Each method gets its own validation-selected threshold and smoothing.",'',
        "The **clip-manual** method uses the same architecture and training budgets on four human-labelled training recordings, one per recipe, with four target units. It was added after the pseudo-supervised model nearly tied the prior on validation and the training audit showed persistent automatic labels. This decision was declared before fresh test prediction or scoring. Both recordings and supervision source differ, so this comparison does not isolate a causal label-quality effect. Its eggs training recording has no separate egg-mixture annotation; that query receives no direct training examples.",'',
        f"Training-recipe transfer: mean F1 {pct(seen['micro_f1'])}. Test-only pizza recipe and cheese/batter objects: mean F1 {pct(pizza['micro_f1'])}. The pizza slice is just one recording with two targets; do not generalize this number to every unseen recipe.",'',
        "## Fresh test units",'',
        '| Video / target | F1 | Top-one hit | Human distinct states | Predicted distinct states | Missed human states |',
        '|---|---:|---:|---:|---:|---|',*units,'',
        "## Held-out state-name experiment",'',
        f"The separate CLIP run withheld **peeled** from supervised train/validation candidates and excluded labelled frames containing it. Evaluation restored all 77 names. Peeled occurs in {len(heldrows)} test unit(s); its mean F1 over those units is {pct(heldf1) if heldf1 is not None else 'not measurable because no positive human labels were present'}.",'',
        "This tests transfer to a held-out labelled state name. Pretrained CLIP/VidOSC and unsupervised temporal context may have seen the appearance, so it is not a claim that the model has never seen peeled vegetables. Custom free-form descriptions are supported inputs but require their own annotated evaluation.",'',
        "## What was changed",'',
        "Your own object-conditioned text scorer takes CLIP visual features, VidOSC temporal context, and an object-name embedding. The motion version adds official SlowFast R50 Kinetics400 features. An own MLP maps these to centered CLIP text embeddings. Independent sigmoid scores allow overlapping states and a variable number of active states. The original four-class output is bypassed. These scores are not calibrated probabilities.",'',
        "The candidates for this evaluation are still the 77 CMU state names. The model selects from them; it does not invent state names or discover an unrestricted state vocabulary. There are no object crops or tracker; full-frame features and an object-name query approximate object conditioning, and VidOSC's object-feature input is zero-filled. The transformer sees the full video and motion windows look ahead, so this is offline analysis rather than causal live streaming.",'',
        "## Verification and limits",'',
        "The fourteen broader tests check frame-clock conversion, dense versus binned metric agreement, interval matching, unknown annotation gaps and annotation tails, held-out label exclusion, subject leakage checks, candidate subset/order consistency, object-query effects, cached context equivalence, shared-query gradient equivalence, and encoder gradients. Nine legacy tests also pass. GPU outputs were checked against CPU implementations on synthetic inputs and real training frames. SlowFast uses verified half precision; CLIP uses full precision after its half-precision check failed.",'',
        "Training pseudo labels can persist when the target is outside the view. The training-only audit sheet records examples; no labels were manually corrected. The primary pseudo-supervised runs reserve manual labels for validation/test; clip-manual uses a separate, subject-disjoint human training set. This training-label noise and missing object localization remain material limitations.",'',
        "The genuine InternVideo comparison remains pending because its official pretrained checkpoint is gated and unavailable in this environment. The adapter is implemented, but no InternVideo scores are claimed. The source-code GitHub link alone does not provide those weights.",'',
        "No teammate source files were edited. No commits, pushes, or pull requests were made. Checkpoints, manifests, predictions, per-unit/per-state metrics, and intervals are saved in this own module. The previous 60.7% salad pilot used different training labels, a narrower split, and different preprocessing, so it is not a like-for-like comparison with this broader test.",'',
        "Test results are now exposed. Changes motivated by these scores need a new held-out split to support a fresh generalization claim.",'',
        "A concrete example of a misleading headline: the fresh egg-mixture target has 100% highest-label hit accuracy, but just 40.1% multi-label F1 and 0% exact label-set accuracy. It has seven human-labelled distinct states, while the selected model predicts 12, missing flipped, in-pan, served, and solid. Correctly ranking one common label does not mean the complete state description is correct.",'',
        "The next experiments should address object localization, normalization and candidate centering for unseen object names, and how never-positive state candidates are treated during training. The current model can produce many extra states for new queries (65 predicted names for pizza batter versus five human-labelled names). Post-evaluation diagnostics are saved separately; they suggest possible contributors without establishing causation. Use fresh recordings for the next generalization test.",'',
        "![Comparison of test metrics](comparison.png)"]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5.5));x=np.arange(len(labels));width=.36
    ax.bar(x-width/2,np.asarray(f1s)*100,width,label='Multi-label F1')
    ax.bar(x+width/2,np.asarray(top1s)*100,width,label='Top-one human-label hit')
    ax.set_ylim(0,100);ax.set_ylabel('Mean across test object/video units (%)')
    ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_title('Fresh CMU test: state detection and highest-score label hit')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=160);plt.close(fig)
    # This is a fixed fresh salad example, not a test example selected for its score.
    import torch
    from metrics import annotation_matrix
    from temporal_states import expand_samples,smooth_scores
    from metrics import intervals
    saved=torch.load(HERE/'results'/best/'test-predictions.pt',weights_only=False)
    key='S15_Salad_7150991-451::vegetables';video=key.split('::')[0];meta=saved['metadata'][video]
    ann=read(DATA/'test_labels.json')[key]
    from protocol import WORKSPACE
    vocab=read(WORKSPACE/'work/chefost-data/MOSCATO/vocabulary/CMU/state_dict.json')
    aliases={k.strip().lower():v.strip().lower() for k,v in vocab.get('all2one',{}).items()}
    truth,valid=annotation_matrix(meta,ann,saved['states'],aliases)
    policy=saved['policies']['trained']
    pred=expand_samples(smooth_scores(saved['methods']['trained'][key],policy['smoothing']),meta['sample_anchors'],
                        meta['raw_frame_count'])>=policy['threshold']
    columns=np.flatnonzero(truth[valid].any(axis=0)|pred[valid].any(axis=0))
    fig,axes=plt.subplots(2,1,figsize=(11,max(5,len(columns)*.5+2)),sharex=True,sharey=True)
    for ax,values,color,title in zip(axes,(truth>0,pred),('#2070a5','#cf742b'),('Human labels','Model predictions')):
        for row,col in enumerate(columns):
            segments=[(a/meta['fps'],(b-a)/meta['fps']) for a,b in intervals(values[:,col]&valid)]
            ax.broken_barh(segments,(row-.35,.7),facecolors=color)
        for a,b in intervals(~valid):ax.axvspan(a/meta['fps'],b/meta['fps'],color='#ededed',zorder=-1)
        ax.set_yticks(np.arange(len(columns)),[saved['states'][c] for c in columns]);ax.set_title(title);ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlabel('Time in seconds; grey areas have no human state labels and are excluded from scoring')
    axes[-1].set_xlim(0,meta['raw_frame_count']/meta['fps']);fig.suptitle('Fresh salad example: overlapping named state intervals')
    fig.tight_layout();fig.savefig(HERE/'salad-timeline.png',dpi=150);plt.close(fig)
    with (HERE/'REPORT.md').open('a',encoding='utf-8') as handle:
        handle.write('\n![Fixed fresh salad timeline example](salad-timeline.png)\n')
    write(HERE/'summary.json',{'validation_selected_model':best,'mean_test_multi_label_f1':m['micro_f1'],
                             'mean_test_top1_label_hit':m['top1_frame_hit_accuracy'],'prior_mean_test_f1':baseline['micro_f1'],
                             'strongest_constant_prior_test_f1':strongest_prior,'accuracy_goal_achieved':False,
                             'heldout_peeled_mean_f1_over_positive_units':heldf1,'internvideo':'pending checkpoint'})
    print('Saved report',HERE/'REPORT.md',flush=True)

if __name__=='__main__':main()
