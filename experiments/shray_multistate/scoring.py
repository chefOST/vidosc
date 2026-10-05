"""Separate multi-label metrics for the new scorer; teammate evaluator untouched."""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from temporal_states import expand_samples, smooth_scores


def label_matrix(sequence, states, aliases=None):
    columns = {state: index for index, state in enumerate(states)}
    aliases = aliases or {}
    target = np.zeros((len(sequence), len(states)), dtype=np.float32)
    valid = np.zeros(len(sequence), dtype=bool)
    for frame, labels in enumerate(sequence):
        for label in labels:
            label = label.strip().lower()
            label = aliases.get(label, label)
            if label not in columns:
                raise ValueError(f'Label {label!r} is outside the scoring vocabulary')
            target[frame, columns[label]] = 1
        valid[frame] = bool(labels)
    return target, valid


def sampled_targets(bundle, annotation, states, aliases):
    metadata = bundle['metadata']
    label_fps = annotation['fps']
    if abs(len(annotation['states'])/label_fps - metadata['raw_frame_count']/metadata['fps']) > 0.2:
        raise ValueError(f'Annotation/video duration mismatch for {metadata["video_id"]}')
    indices = [min(int(anchor * label_fps / metadata['fps']), len(annotation['states'])-1)
               for anchor in metadata['sample_anchors']]
    target, valid = label_matrix([annotation['states'][index] for index in indices], states, aliases)
    return torch.from_numpy(target), torch.from_numpy(valid)


def evaluate_scores(scores, metadata, annotation, states, aliases, threshold, smoothing=1, include_details=True):
    if annotation['fps'] != metadata['fps'] or len(annotation['states']) != metadata['raw_frame_count']:
        raise ValueError('Evaluation annotations must match raw video frame coordinates exactly')
    scores = smooth_scores(scores, smoothing)
    dense = expand_samples(scores, metadata['sample_anchors'], metadata['raw_frame_count'])
    truth, valid = label_matrix(annotation['states'], states, aliases)
    if not valid.any():
        raise ValueError('No annotated frames available to score')
    truth = truth[valid]
    dense = dense[valid]
    predictions = dense >= threshold
    true_binary = truth.astype(bool)
    tp = int((true_binary & predictions).sum())
    fp = int((~true_binary & predictions).sum())
    fn = int((true_binary & ~predictions).sum())
    p = tp / max(tp+fp,1)
    r = tp / max(tp+fn,1)
    f1 = 2*tp / max(2*tp+fp+fn,1)
    intersection = (truth.astype(bool) & predictions).sum(axis=1)
    union = (truth.astype(bool) | predictions).sum(axis=1)
    top1 = dense.argmax(axis=1)
    metrics = {
        'evaluated_frames': int(valid.sum()), 'skipped_empty_gt_frames': int((~valid).sum()),
        'micro_precision':float(p), 'micro_recall':float(r), 'micro_f1':float(f1),
        'top1_frame_hit_accuracy':float(truth[np.arange(len(truth)),top1].mean()),
        'exact_state_set_accuracy':float(np.all(truth == predictions, axis=1).mean()),
        'mean_frame_jaccard':float((intersection / np.maximum(union,1)).mean()),
        'threshold':float(threshold), 'smoothing_samples':int(smoothing),
        'prediction_unit':'one independent score per candidate state; simultaneous states allowed',
    }
    if include_details:
        present = truth.sum(axis=0) > 0
        metrics['map_over_gt_present_states'] = float(average_precision_score(truth[:,present],dense[:,present],average='macro'))
        pc,rc,fc,support = precision_recall_fscore_support(truth,predictions,average=None,zero_division=0)
        metrics['per_state'] = {label:{'precision':float(pc[i]),'recall':float(rc[i]),'f1':float(fc[i]),
            'gt_positive_frames':int(support[i]),'predicted_positive_frames':int(predictions[:,i].sum())}
            for i,label in enumerate(states) if support[i] or predictions[:,i].any()}
        gt_states = {states[i] for i in np.flatnonzero(present)}
        pred_states = {states[i] for i in np.flatnonzero(predictions.any(axis=0))}
        metrics['gt_unique_states'] = sorted(gt_states)
        metrics['predicted_unique_states_on_annotated_frames'] = sorted(pred_states)
        metrics['unique_state_count_absolute_error'] = abs(len(gt_states)-len(pred_states))
        metrics['state_set_precision'] = len(gt_states & pred_states)/max(len(pred_states),1)
        metrics['state_set_recall'] = len(gt_states & pred_states)/len(gt_states)
    return metrics


def choose_validation_policy(scores_by_video, bundles, annotations, states, aliases):
    best = None
    for smoothing in (1,3,5):
        for threshold in np.linspace(0.1,0.9,17):
            metrics = [evaluate_scores(scores_by_video[v],bundles[v]['metadata'],annotations[v],states,aliases,
                                       threshold,smoothing,False) for v in scores_by_video]
            # The pilot has one validation video. Macro video mean is explicit
            # for extension; no held-out test annotation enters policy selection.
            f1 = float(np.mean([item['micro_f1'] for item in metrics]))
            precision = float(np.mean([item['micro_precision'] for item in metrics]))
            choice = {'micro_f1':f1,'micro_precision':precision,'threshold':float(threshold),'smoothing':smoothing}
            if best is None or (f1,precision) > (best['micro_f1'],best['micro_precision']):
                best = choice
    return best
