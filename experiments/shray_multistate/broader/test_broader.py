import unittest,sys,copy
from pathlib import Path
import numpy as np
import torch
import protocol
sys.path.insert(0,str(protocol.HERE))
from protocol import assert_disjoint,read,CONFIG
from model import QueryStateModel
from training_targets import label_matrix,sampled_targets
from temporal_states import expand_samples

class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.states=['raw','cut','peeled'];self.meta={'fps':2,'raw_frame_count':8,'sample_anchors':[0,3,6]}
        self.ann={'fps':2,'states':[['raw'],['raw','peeled'],[],['cut'],['cut'],['cut','peeled'],['raw'],['peeled']]}
    def test_pseudo_to_raw_clock_alignment(self):
        bundle={'metadata':{'fps':30,'raw_frame_count':180,'sample_anchors':[0,60,120],'video_id':'example'}}
        ann={'fps':10,'states':[['raw']]*20+[['cut']]*20+[['peeled']]*20}
        y,mask=sampled_targets(bundle,ann,self.states,{})
        self.assertTrue(mask.all());self.assertTrue(torch.equal(y,torch.eye(3)))
        with self.assertRaises(ValueError):sampled_targets(bundle,{**ann,'states':ann['states'][:30]},self.states,{})
    def test_split_rejects_subject_overlap(self):
        cfg=read(CONFIG);assert_disjoint(cfg)
        cfg=copy.deepcopy(cfg);cfg['test'][0]='S07_Salad_another'
        with self.assertRaises(ValueError):assert_disjoint(cfg)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8);torch.manual_seed(3)
        cls.model=QueryStateModel().eval()
    def test_candidate_count_and_order_preserve_scores(self):
        visual=torch.randn(7,768);query=torch.randn(768);texts=torch.randn(77,768)
        with torch.no_grad():
            all_scores=self.model(visual,query,texts)
            chosen=self.model(visual,query,texts[[2,50,9]])
        self.assertEqual(tuple(chosen.shape),(7,3))
        torch.testing.assert_close(chosen,all_scores[:,[2,50,9]],rtol=1e-5,atol=1e-5)
    def test_query_changes_scores(self):
        visual=torch.randn(7,768);texts=torch.randn(5,768)
        with torch.no_grad():
            one=self.model(visual,torch.zeros(768),texts)
            two=self.model(visual,torch.ones(768),texts)
        self.assertGreater((one-two).abs().max().item(),.01)
    def test_cached_context_matches_encoder_forward(self):
        visual=torch.randn(7,768);query=torch.randn(768);texts=torch.randn(5,768)
        with torch.no_grad():
            temporal=self.model.encoder(torch.cat([visual,torch.zeros_like(visual)],dim=-1)[None],middle=True)[0]
            cached=self.model(visual,query,texts,temporal=temporal)
            actual=self.model(visual,query,texts)
        torch.testing.assert_close(cached,actual)
    def test_encoder_freeze_then_finetune_has_gradients(self):
        model=QueryStateModel().train();visual=torch.randn(7,768);query=torch.randn(768);texts=torch.randn(3,768)
        model(visual,query,texts).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        model.zero_grad();model.train_encoder=True;model.encoder.requires_grad_(True)
        model(visual,query,texts).sum().backward()
        grads=[p.grad for p in model.encoder.parameters() if p.grad is not None]
        self.assertTrue(grads);self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads),0)
    def test_shared_video_context_preserves_two_query_gradients(self):
        model=QueryStateModel(train_encoder=True).train();model.head.projection[2].p=0
        visual=torch.randn(7,768);queries=[torch.randn(768),torch.randn(768)];texts=torch.randn(5,768)
        sum(model(visual,q,texts).sum() for q in queries).backward()
        reference={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
        model.zero_grad();context=model.context(visual)
        sum(model(visual,q,texts,temporal=context).sum() for q in queries).backward()
        for key,p in model.named_parameters():
            if key in reference:torch.testing.assert_close(p.grad,reference[key],rtol=1e-4,atol=1e-4)

if __name__=='__main__':unittest.main()
