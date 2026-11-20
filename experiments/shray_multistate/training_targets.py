"""Encode training labels on feature anchors; no benchmark metrics."""
import numpy as np
import torch

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
