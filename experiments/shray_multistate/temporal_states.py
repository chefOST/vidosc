"""Text-conditioned multi-label scoring, independent of a fixed state count."""

import numpy as np
import torch
from torch import nn


class TextStateHead(nn.Module):
    """Learn alignment to text space; candidate count is a forward-pass input."""

    def __init__(self, input_dim, text_dim=768, hidden_dim=256):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))
        self.register_buffer('text_center', torch.zeros(text_dim))
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                        nn.Dropout(0.15), nn.Linear(hidden_dim, text_dim))
        self.log_scale = nn.Parameter(torch.tensor(np.log(10), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, features, candidate_embeddings):
        normalized = (features - self.feature_mean) / self.feature_std
        visual = nn.functional.normalize(self.projection(normalized), dim=-1)
        texts = nn.functional.normalize(candidate_embeddings - self.text_center, dim=-1)
        return visual @ texts.T * self.log_scale.clamp(0, np.log(100)).exp() + self.bias

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Earlier local prototypes used uncentered text. Preserve their outputs
        # while adding the optional fixed training-vocabulary center.
        if 'text_center' not in state_dict:
            state_dict = dict(state_dict)
            state_dict['text_center'] = torch.zeros_like(self.text_center)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)


class MultiStateVidOSC(nn.Module):
    """End-to-end temporal scorer for pre-extracted video features and state text.

    The original four-class head is bypassed. An own trained text head returns
    [batch, time, number_of_candidates] logits, with no fixed state order.
    Image/text extraction remains a separate frozen dependency.
    """

    def __init__(self, temporal_encoder, text_head, train_encoder=False, encoder_dropout=True):
        super().__init__()
        self.encoder = temporal_encoder.eval()
        self.train_encoder = train_encoder
        self.encoder_dropout = encoder_dropout
        self.encoder.requires_grad_(train_encoder)
        self.text_head = text_head

    def forward(self, visual_features, candidate_embeddings, object_features=None):
        if visual_features.ndim != 3 or visual_features.shape[-1] != 768:
            raise ValueError('Visual features must have shape [batch, time, 768]')
        if object_features is None:
            object_features = torch.zeros_like(visual_features)
        if object_features.shape != visual_features.shape:
            raise ValueError('Object features must align with the visual timeline')
        if self.train_encoder and self.training:
            self.encoder.train(self.encoder_dropout)
            context = self.encoder(torch.cat([visual_features,object_features],dim=-1),middle=True)
        else:
            self.encoder.eval()
            with torch.no_grad():
                context = self.encoder(torch.cat([visual_features,object_features],dim=-1),middle=True)
        inputs = torch.cat([visual_features,context],dim=-1)
        return self.text_head(inputs,candidate_embeddings)


def select_features(bundle, mode):
    if mode == 'visual':
        return bundle['visual'].float()
    if mode == 'temporal':
        return bundle['temporal'].float()
    if mode == 'joint':
        return torch.cat([bundle['visual'], bundle['temporal']], dim=-1).float()
    raise ValueError(f'Unknown feature mode {mode}')


def smooth_scores(scores, width):
    values = np.asarray(scores)
    if width < 1 or width % 2 == 0:
        raise ValueError('Smoothing width must be a positive odd integer')
    if values.ndim != 2 or not len(values):
        raise ValueError('Scores must be a nonempty time-by-state matrix')
    pad = width // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode='edge')
    return np.median(np.stack([padded[start:start+len(values)] for start in range(width)]), axis=0)


def expand_samples(values, anchors, num_frames):
    values = np.asarray(values)
    anchors = list(anchors)
    if (not anchors or anchors[0] != 0 or len(values) != len(anchors)
            or any(a >= b for a,b in zip(anchors, anchors[1:]))
            or anchors[-1] >= num_frames):
        raise ValueError('Anchors must partition all frames from zero without duplicates')
    widths = np.diff(anchors + [num_frames])
    return np.repeat(values, widths, axis=0)


def state_intervals(active, labels, anchors, num_frames, fps):
    active = np.asarray(active, dtype=bool)
    if active.shape != (len(anchors), len(labels)) or len(set(labels)) != len(labels):
        raise ValueError('Labels and anchors must match the activity matrix')
    # Also validate the partition before constructing interval coordinates.
    expand_samples(active, anchors, num_frames)
    result = []
    for column, label in enumerate(labels):
        start = None
        for timestep in range(len(active)+1):
            present = timestep < len(active) and active[timestep, column]
            if present and start is None:
                start = timestep
            elif not present and start is not None:
                first = anchors[start]
                end = anchors[timestep] if timestep < len(active) else num_frames
                result.append({'state': label, 'start_frame': first, 'end_frame_exclusive': end,
                               'start_s': first / fps, 'end_s_exclusive': end / fps})
                start = None
    return sorted(result, key=lambda item: (item['start_frame'], item['state']))
