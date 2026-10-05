"""Balanced state metrics and one-to-one segment matching on known annotation frames."""
import numpy as np
from protocol import HERE
from scoring import label_matrix
from temporal_states import smooth_scores,expand_samples

def counts_from_arrays(scores,truth,valid):
    pred=np.asarray(scores,dtype=bool)[valid];target=truth[valid].astype(bool)
    tp=(pred&target).sum(axis=0);fp=(pred&~target).sum(axis=0);fn=(~pred&target).sum(axis=0)
    return tp,fp,fn

def f1_counts(tp,fp,fn):
    return np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(np.asarray(tp,dtype=float)),where=(2*tp+fp+fn)>0)

def annotation_matrix(meta,annotation,states,aliases):
    if annotation['fps']!=meta['fps'] or len(annotation['states'])>meta['raw_frame_count']:
        raise ValueError('Raw annotation clock mismatch')
    truth,valid=label_matrix(annotation['states'],states,aliases)
    # Published frame arrays start at zero. Coverage beyond their end is unknown.
    missing=meta['raw_frame_count']-len(truth)
    if missing:
        truth=np.pad(truth,((0,missing),(0,0)));valid=np.pad(valid,(0,missing),constant_values=False)
    return truth,valid

class ValidationCounts:
    """Collapse reference frames into prediction bins; preserve exact raw-frame micro-F1."""
    def __init__(self,meta,annotation,states,aliases,columns=None,ignore_states=()):
        truth,valid=annotation_matrix(meta,annotation,states,aliases)
        ignore=[states.index(s) for s in ignore_states]
        if ignore:valid &= ~truth[:,ignore].any(axis=1)
        if columns is not None:truth=truth[:,columns]
        anchors=meta['sample_anchors'];bins=np.repeat(np.arange(len(anchors)),np.diff(anchors+[meta['raw_frame_count']]))
        self.positives=np.zeros((len(anchors),truth.shape[1]),dtype=np.float64)
        np.add.at(self.positives,bins,truth*valid[:,None])
        self.known=np.bincount(bins,weights=valid,minlength=len(anchors))[:,None]
        self.support=self.positives.sum(axis=0)
    def measure(self,pred):
        tp=(pred*self.positives).sum(axis=0)
        fp=(pred*self.known).sum(axis=0)-tp
        fn=self.support-tp
        t,f,n=tp.sum(),fp.sum(),fn.sum()
        return {'micro_f1':float(2*t/max(2*t+f+n,1)), 'precision':float(t/max(t+f,1)),
                'macro_f1_gt_present':float(f1_counts(tp,fp,fn)[self.support>0].mean()) if (self.support>0).any() else 0}

def choose_policy(scores,counts):
    best=None
    for smoothing in (1,3,5):
        smoothed={key:smooth_scores(value,smoothing) for key,value in scores.items()}
        for threshold in np.linspace(.05,.95,19):
            rows=[counts[key].measure(value>=threshold) for key,value in smoothed.items() if counts[key].known.sum()>0]
            if not rows:raise ValueError('No validation frames')
            result={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            result.update(threshold=float(threshold),smoothing=smoothing,selection_unit='equal weight per object/video unit')
            if best is None or (result['micro_f1'],result['precision'])>(best['micro_f1'],best['precision']):best=result
    if best is None:raise ValueError('No validation frames')
    return best

def intervals(mask):
    padded=np.r_[False,np.asarray(mask,dtype=bool),False].astype(int)
    edges=np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(),edges[1::2].tolist()))

def segment_match(predicted,reference,threshold):
    """Maximum-cardinality same-label matching; one predicted segment cannot match twice."""
    adjacency=[];ious={}
    for i,(a,b) in enumerate(predicted):
        hits=[]
        for j,(c,d) in enumerate(reference):
            overlap=max(0,min(b,d)-max(a,c));union=max(b,d)-min(a,c)
            iou=overlap/max(union,1)
            if overlap and iou>=threshold:hits.append(j);ious[i,j]=iou
        adjacency.append(sorted(hits,key=lambda j:ious[i,j],reverse=True))
    assigned={}
    def augment(i,seen):
        for j in adjacency[i]:
            if j in seen:continue
            seen.add(j)
            if j not in assigned or augment(assigned[j],seen):assigned[j]=i;return True
        return False
    for i in range(len(predicted)):augment(i,set())
    return [(i,j) for j,i in assigned.items()]

def evaluate(scores,meta,annotation,states,aliases,policy,heldout_states=()):
    truth,valid=annotation_matrix(meta,annotation,states,aliases)
    if not valid.any():raise ValueError('No known annotation frames')
    dense=expand_samples(smooth_scores(scores,policy['smoothing']),meta['sample_anchors'],meta['raw_frame_count'])
    pred=dense>=policy['threshold'];tp,fp,fn=counts_from_arrays(pred,truth,valid)
    t,f,n=int(tp.sum()),int(fp.sum()),int(fn.sum());support=truth[valid].sum(axis=0)
    top=dense[valid].argmax(axis=1);known_truth=truth[valid]
    per_f1=f1_counts(tp,fp,fn)
    gt_names=[states[i] for i in np.flatnonzero(support)]
    pred_names=[states[i] for i in np.flatnonzero(pred[valid].any(axis=0))]
    intersections=(pred[valid]&known_truth.astype(bool)).sum(axis=1)
    unions=(pred[valid]|known_truth.astype(bool)).sum(axis=1)
    result={'evaluated_frames':int(valid.sum()),'skipped_empty_gt_frames':int((~valid).sum()),
            'published_annotation_array_frames':len(annotation['states']),
            'trailing_frames_outside_published_annotation_coverage':meta['raw_frame_count']-len(annotation['states']),
            'micro_precision':t/max(t+f,1),'micro_recall':t/max(t+n,1),'micro_f1':2*t/max(2*t+f+n,1),
            'macro_f1_gt_present':float(per_f1[support>0].mean()),
            'top1_frame_hit_accuracy':float(known_truth[np.arange(len(top)),top].mean()),
            'balanced_top1_label_recall':float(np.mean([((top==i)&(known_truth[:,i]>0)).sum()/support[i] for i in np.flatnonzero(support)])),
            'exact_state_set_accuracy':float((pred[valid]==known_truth).all(axis=1).mean()),
            'mean_frame_jaccard':float((intersections/np.maximum(unions,1)).mean()),
            'gt_unique_states':gt_names,'predicted_unique_states_on_annotated_frames':pred_names,
            'unique_state_count_absolute_error':abs(len(gt_names)-len(pred_names)),
            'state_set_precision':len(set(gt_names)&set(pred_names))/max(len(pred_names),1),
            'state_set_recall':len(set(gt_names)&set(pred_names))/max(len(gt_names),1),
            'missed_state_names':sorted(set(gt_names)-set(pred_names)),'extra_state_names':sorted(set(pred_names)-set(gt_names)),
            'counts':{'tp':t,'fp':f,'fn':n},'per_state':{}}
    segments={x:{'tp':0,'fp':0,'fn':0} for x in (.1,.25,.5)};endpoint_errors=[]
    for i,state in enumerate(states):
        if support[i] or pred[valid,i].any():
            result['per_state'][state]={'precision':float(tp[i]/max(tp[i]+fp[i],1)),'recall':float(tp[i]/max(tp[i]+fn[i],1)),
                                        'f1':float(per_f1[i]),'gt_positive_frames':int(support[i]),'predicted_positive_frames':int(pred[valid,i].sum())}
        reference=intervals((truth[:,i]>0)&valid);predicted=intervals(pred[:,i]&valid)
        for threshold,totals in segments.items():
            matched=segment_match(predicted,reference,threshold)
            totals['tp']+=len(matched);totals['fp']+=len(predicted)-len(matched);totals['fn']+=len(reference)-len(matched)
            if threshold==.5:
                for a,b in matched:
                    endpoint_errors.extend([abs(predicted[a][0]-reference[b][0])/meta['fps'],abs(predicted[a][1]-reference[b][1])/meta['fps']])
    result['segment_metrics']={str(k):{**v,'f1':2*v['tp']/max(2*v['tp']+v['fp']+v['fn'],1)} for k,v in segments.items()}
    result['matched_segment_endpoint_mae_seconds_iou_0.5']=float(np.mean(endpoint_errors)) if endpoint_errors else None
    result['heldout_states']={s:result['per_state'].get(s,{'precision':0,'recall':0,'f1':0,'gt_positive_frames':0,'predicted_positive_frames':0}) for s in heldout_states}
    return result

def aggregate(rows):
    keys=('top1_frame_hit_accuracy','balanced_top1_label_recall','micro_precision','micro_recall','micro_f1',
          'macro_f1_gt_present','exact_state_set_accuracy','mean_frame_jaccard','unique_state_count_absolute_error',
          'state_set_precision','state_set_recall')
    result={'object_video_units':len(rows),'macro_over_object_video_units':{k:float(np.mean([r[k] for r in rows.values()])) for k in keys}}
    tp=sum(r['counts']['tp'] for r in rows.values());fp=sum(r['counts']['fp'] for r in rows.values());fn=sum(r['counts']['fn'] for r in rows.values())
    result['pooled_micro_f1']=2*tp/max(2*tp+fp+fn,1)
    result['segment_f1_macro_over_units']={str(k):float(np.mean([r['segment_metrics'][str(k)]['f1'] for r in rows.values()])) for k in (.1,.25,.5)}
    return result
