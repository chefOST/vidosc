"""Check the coordinate convention required by the original detector NMS."""
import torch
from moscato.hand_object import inclusive_nms


def test_nms_counts_pixel_endpoints():
    # Inclusive IoU is 1/3, continuous IoU is zero: catches a silent convention change.
    boxes = torch.tensor([[0., 0., 1., 1.], [1., 0., 2., 1.], [9., 9., 10., 10.]])
    keep = inclusive_nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.3)
    assert keep.tolist() == [0, 2]
    assert boxes[0].tolist() == [0., 0., 1., 1.]


def test_empty_nms():
    assert inclusive_nms(torch.empty((0, 4)), torch.empty(0), 0.3).numel() == 0
