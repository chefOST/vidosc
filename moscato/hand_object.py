"""Inference adapter for ddshan/hand_object_detector, using torchvision ops.

Uses original network and bbox transforms. Full numerical parity with the old
compiled extension is unverified. Does not select crops or use state labels.
"""
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import RoIAlign, RoIPool, nms, roi_align, roi_pool


def inclusive_nms(boxes, scores, threshold):
    """Original NMS counts both endpoints (+1 width/height)."""
    continuous = boxes.clone()
    continuous[:, 2:] += 1
    return nms(continuous, scores, threshold)


def load_detector(source, checkpoint, device):
    source = Path(source)
    if 'model' in sys.modules:
        raise RuntimeError('Run detector in a separate process from other model packages')
    sys.path.insert(0, str(source / 'lib'))
    # Original ROIAlign has no half-pixel shift (torchvision aligned=False).
    ops = types.ModuleType('model.roi_layers')
    ops.ROIAlign, ops.ROIPool = RoIAlign, RoIPool
    ops.roi_align, ops.roi_pool, ops.nms = roi_align, roi_pool, inclusive_nms
    sys.modules['model.roi_layers'] = ops
    from model.utils.config import cfg
    from model.faster_rcnn.resnet import resnet
    # These are the inference-relevant values in upstream cfgs/res101.yml.
    cfg.POOLING_SIZE = 7
    cfg.POOLING_MODE = 'align'
    cfg.CROP_RESIZE_WITH_MAX_POOL = False
    cfg.TEST.HAS_RPN = True
    cfg.ANCHOR_SCALES = [8, 16, 32, 64]
    cfg.ANCHOR_RATIOS = [0.5, 1, 2]
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg.POOLING_MODE = data.get('pooling_mode', 'align')
    if cfg.POOLING_MODE != 'align':
        raise ValueError('Only the released align checkpoint is supported')
    net = resnet(np.asarray(['__background__', 'targetobject', 'hand']),
                 101, pretrained=False, class_agnostic=False)
    net.create_architecture()
    net.load_state_dict(data['model'], strict=True)
    # Upstream overrides train() without returning self; eval() returns None.
    net.eval()
    net.to(device)
    return net, cfg


@torch.inference_mode()
def detect_frame(net, cfg, bgr, device, threshold=0.5):
    from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0, 1]')
    height, width = bgr.shape[:2]
    pixels = bgr.astype(np.float32) - cfg.PIXEL_MEANS.astype(np.float32)
    scale = float(cfg.TEST.SCALES[0]) / min(height, width)
    if np.round(scale * max(height, width)) > cfg.TEST.MAX_SIZE:
        scale = float(cfg.TEST.MAX_SIZE) / max(height, width)
    pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
    info = torch.tensor([[pixels.shape[0], pixels.shape[1], scale]], device=device)
    output = net(image, info, torch.zeros((1, 1, 5), device=device),
                 torch.zeros(1, dtype=torch.long, device=device),
                 torch.zeros((1, 1, 5), device=device))
    rois, probabilities, deltas = output[:3]
    ext = output[-1]
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        deltas = (deltas.reshape(-1, 4) * deltas.new_tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
                  + deltas.new_tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)).reshape(1, -1, 12)
    boxes = clip_boxes(bbox_transform_inv(rois[:, :, 1:5], deltas, 1), info, 1)[0] / scale
    scores = probabilities[0]
    contacts = ext[0][0][0].argmax(-1)
    offsets = ext[1][0][0]
    sides = (ext[2][0][0].sigmoid() > 0.5).flatten()
    detections = []
    for cls_id, name in [(1, 'targetobject'), (2, 'hand')]:
        ids = torch.where(scores[:, cls_id] > threshold)[0]
        selected = boxes[ids, cls_id * 4:(cls_id + 1) * 4]
        keep = inclusive_nms(selected, scores[ids, cls_id], cfg.TEST.NMS)
        for index in ids[keep].tolist():
            box = boxes[index, cls_id * 4:(cls_id + 1) * 4].clone()
            box[0::2].clamp_(0, width - 1)
            box[1::2].clamp_(0, height - 1)
            if not torch.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            detection = {'class': name, 'score': float(scores[index, cls_id]),
                         'box_xyxy': box.cpu().tolist()}
            if cls_id == 2:
                detection.update(contact_state_index=int(contacts[index]),
                                 right_hand=bool(sides[index]),
                                 offset=offsets[index].cpu().tolist())
            detections.append(detection)
    return detections
