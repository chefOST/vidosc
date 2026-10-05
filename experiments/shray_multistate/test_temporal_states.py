import unittest

import numpy as np
import torch

from temporal_states import TextStateHead, MultiStateVidOSC, expand_samples, smooth_scores, state_intervals
from scoring import evaluate_scores


class TemporalStateTests(unittest.TestCase):
    def test_old_uncentered_checkpoints_keep_their_scores(self):
        torch.manual_seed(11)
        original=TextStateHead(16,8,16).eval()
        legacy={k:v for k,v in original.state_dict().items() if k!='text_center'}
        restored=TextStateHead(16,8,16).eval()
        restored.text_center.fill_(1)
        restored.load_state_dict(legacy,strict=True)
        x,text=torch.randn(4,16),torch.randn(6,8)
        with torch.no_grad():torch.testing.assert_close(original(x,text),restored(x,text))

    def test_encoder_can_be_finetuned_in_own_model(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj=torch.nn.Linear(1536,512)
            def forward(self,x,middle=False):
                if not middle: raise AssertionError('Fixed head must be bypassed')
                return self.proj(x)
        encoder=Encoder()
        model=MultiStateVidOSC(encoder,TextStateHead(1280,8,16),train_encoder=True,encoder_dropout=False)
        model.train()
        model(torch.randn(1,6,768),torch.randn(9,8)).sum().backward()
        self.assertTrue(encoder.proj.weight.requires_grad)
        self.assertFalse(encoder.training)
        self.assertGreater(float(encoder.proj.weight.grad.abs().sum()),0)

    def test_temporal_encoder_is_used_and_frozen(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale=torch.nn.Parameter(torch.ones(1))
                self.calls=0
            def forward(self,x,middle=False):
                self.calls+=1
                if not middle: raise AssertionError('Fixed head must be bypassed')
                return x[...,:512]*self.scale
        encoder=Encoder()
        head=TextStateHead(1280,8,16)
        model=MultiStateVidOSC(encoder,head)
        logits=model(torch.randn(1,6,768),torch.randn(9,8))
        self.assertEqual(logits.shape,(1,6,9))
        self.assertEqual(encoder.calls,1)
        self.assertFalse(encoder.scale.requires_grad)
        logits.sum().backward()
        self.assertIsNotNone(head.projection[0].weight.grad)
        self.assertIsNone(encoder.scale.grad)

    def test_candidate_count_can_change_without_retraining(self):
        torch.manual_seed(42)
        model = TextStateHead(16,8,16).eval()
        x,text = torch.randn(7,16),torch.randn(11,8)
        model.text_center.copy_(text.mean(dim=0))
        with torch.no_grad():
            full = model(x,text)
            subset = model(x,text[[1,5,9]])
        self.assertEqual(full.shape,(7,11))
        self.assertEqual(subset.shape,(7,3))
        torch.testing.assert_close(subset,full[:,[1,5,9]])

    def test_simultaneous_states_recurrence_and_tail(self):
        active = [[1,1,0,0,0],[1,0,1,0,0],[0,0,0,1,1],[1,0,0,0,0]]
        result = state_intervals(active,['whole','peeled','chopped','mixed','coated'],[0,60,120,180],205,30)
        self.assertEqual(len({r['state'] for r in result}),5)
        whole = [r for r in result if r['state']=='whole']
        self.assertEqual(len(whole),2)
        self.assertEqual(whole[-1]['end_frame_exclusive'],205)
        self.assertAlmostEqual(whole[-1]['end_s_exclusive'],205/30)
        self.assertEqual(len([r for r in result if r['start_frame']==0]),2)

    def test_missing_intervals_and_bad_anchors_are_rejected(self):
        for anchors in ([1,3],[0,0],[0,8]):
            with self.assertRaises(ValueError):
                expand_samples(np.zeros((2,2)),anchors,8)

    def test_multilabel_metrics_penalize_missing_colabel(self):
        metadata={'fps':30,'raw_frame_count':4,'sample_anchors':[0,2]}
        labels={'fps':30,'states':[['peeled','chopped'],['peeled','chopped'],['chopped'],[]]}
        metrics=evaluate_scores([[0.9,0.1],[0.1,0.9]],metadata,labels,['peeled','chopped'],{},0.5)
        self.assertEqual(metrics['evaluated_frames'],3)
        self.assertEqual(metrics['top1_frame_hit_accuracy'],1)
        self.assertLess(metrics['micro_f1'],1)
        self.assertAlmostEqual(metrics['exact_state_set_accuracy'],1/3)

    def test_alignment_learns_a_multilabel_mapping(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        model=TextStateHead(4,4,16)
        model.train()
        x=torch.eye(4)
        texts=torch.eye(4)
        targets=torch.tensor([[1.,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1]])
        optimizer=torch.optim.Adam(model.parameters(),lr=0.02)
        loss_fn=torch.nn.BCEWithLogitsLoss()
        initial=float(loss_fn(model(x,texts),targets).detach())
        for _ in range(100):
            optimizer.zero_grad()
            loss=loss_fn(model(x,texts),targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits=model(x,texts)
            final=float(loss_fn(logits,targets))
        self.assertLess(final,initial/2)
        self.assertTrue(torch.equal(logits.sigmoid()>=0.5,targets.bool()))

    def test_smoothing_does_not_impose_state_order(self):
        result=smooth_scores([[0,1],[1,0],[0,1]],3)
        self.assertEqual(result.tolist(),[[0,1],[0,1],[0,1]])
        with self.assertRaises(ValueError):smooth_scores([[1]],2)


if __name__=='__main__':
    unittest.main()
