"""Check query robustness, candidate invariance and raw-frame loss alignment."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'broader'))
import protocol
from models import StableQueryModel,prototype_scores
from metrics import ValidationCounts

class AccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.model=StableQueryModel().eval()
    def test_stable_query_scale_independent_of_training_object_distribution(self):
        one=torch.randn(12,2048);two=torch.randn(12,2048)*20+10
        self.model.initialize_statistics(one)
        a=self.model.head.feature_mean[1280:].clone();b=self.model.head.feature_std[1280:].clone()
        self.model.initialize_statistics(two)
        torch.testing.assert_close(a,self.model.head.feature_mean[1280:])
        torch.testing.assert_close(b,self.model.head.feature_std[1280:])
    def test_stable_candidate_subset_and_order_preserve_actual_logits(self):
        self.model.initialize_statistics(torch.randn(12,2048))
        v=torch.randn(7,768);q=torch.nn.functional.normalize(torch.randn(768),dim=0);c=torch.randn(77,768)
        with torch.inference_mode():
            full=self.model(v,q,c);part=self.model(v,q,c[[5,2,60]])
        torch.testing.assert_close(part,full[:,[5,2,60]],rtol=1e-5,atol=1e-5)
    def test_soft_bin_loss_equals_dense_known_frame_loss(self):
        meta={'fps':1,'raw_frame_count':6,'sample_anchors':[0,3]}
        ann={'fps':1,'states':[['raw'],['raw','cut'],[],['cut'],['cut'],[]]}
        counts=ValidationCounts(meta,ann,['raw','cut'],{})
        logits=torch.tensor([[.1,.8],[-.5,.4]])
        target=torch.tensor(counts.positives/np.maximum(counts.known,1),dtype=torch.float32)
        weights=torch.tensor(counts.known[:,0],dtype=torch.float32)
        binned=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none').mean(1)*weights).sum()/weights.sum()
        dense_logits=logits[[0,0,1,1]]
        dense_target=torch.tensor([[1.,0],[1,1],[0,1],[0,1]])
        dense=torch.nn.functional.binary_cross_entropy_with_logits(dense_logits,dense_target)
        torch.testing.assert_close(binned,dense)
    def test_prototype_candidate_remapping_and_unknown_text_fallback(self):
        b={'visual':torch.zeros(3,768),'temporal':torch.zeros(3,512)}
        entry={'prior':torch.tensor([.2,.7]),'positive':torch.zeros(2,1280),'negative':torch.zeros(2,1280)}
        fit={'states':['raw','cut'],'mean':torch.zeros(1280),'std':torch.ones(1280),'global':entry,'by_object':{}}
        full=prototype_scores(b,'new object',fit,0)
        np.testing.assert_allclose(full,np.tile([.2,.7],(3,1)),rtol=1e-6)
        subset=prototype_scores(b,'new object',fit,12,['cut'])
        np.testing.assert_allclose(subset,full[:,[1]])
        fallback=np.full((3,2),.42)
        result=prototype_scores(b,'new object',fit,12,['novel','raw'],fallback)
        np.testing.assert_allclose(result[:,0],.42)
        np.testing.assert_allclose(result[:,1],full[:,0])
        with self.assertRaises(ValueError):prototype_scores(b,'new object',fit,0,['novel'])

if __name__=='__main__':unittest.main()
