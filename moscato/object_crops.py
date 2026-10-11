"""Declared exploratory crop policy; not claimed to be VidOSC's original recipe."""
import math

POLICY = {
    'name': 'midpoint_hand_associated_union_v1',
    'detection': 'one midpoint frame per one-second window; confidence > 0.5',
    'association': 'upstream filter_object: contact > 0, nearest object center to hand offset endpoint',
    'crop': 'union of associated object boxes, fixed across the eight sampled frames; no extra padding',
    'missing': 'zero object feature, as in upstream ChangeItFeatDataset.load_feat',
    'selection_uses_ground_truth': False,
    'original_crop_recipe_verified': False,
}


def choose_crop(detections, width, height):
    objects = [d for d in detections if d['class'] == 'targetobject']
    hands = [d for d in detections if d['class'] == 'hand' and d['contact_state_index'] > 0]
    if not objects or not hands:
        return None
    chosen = []
    for hand in hands:
        b = hand['box_xyxy']
        mag, dx, dy = hand['offset']
        endpoint = ((b[0]+b[2])/2 + mag*10000*dx, (b[1]+b[3])/2 + mag*10000*dy)
        def distance(obj):
            q = obj['box_xyxy']
            return ((q[0]+q[2])/2-endpoint[0])**2 + ((q[1]+q[3])/2-endpoint[1])**2
        chosen.append(min(objects, key=distance)['box_xyxy'])
    # Convert inclusive detector endpoints to a half-open NumPy crop.
    x1 = max(0, math.floor(min(b[0] for b in chosen)))
    y1 = max(0, math.floor(min(b[1] for b in chosen)))
    x2 = min(width, math.ceil(max(b[2] for b in chosen)) + 1)
    y2 = min(height, math.ceil(max(b[3] for b in chosen)) + 1)
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None
