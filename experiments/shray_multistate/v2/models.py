"""Stable text scoring and supervised visual-temporal prototype calibration."""
import math
import torch
from model import QueryStateModel

class StableQueryModel(QueryStateModel):
    def initialize_statistics(self, features):
        self.head.feature_mean.copy_(features.mean(0))
        self.head.feature_std.copy_(features.std(0).clamp_min(.01))
        # CLIP queries are unit vectors. Use a fixed scale independent of the
        # small set of training object names, avoiding coordinate-wise OOD gain.
        self.head.feature_mean[1280:2048].zero_()
        self.head.feature_std[1280:2048].fill_(1 / math.sqrt(768))

    def forward(self, visual, query, candidates, motion=None, temporal=None):
        features=self.features(visual,query,motion,temporal)
        normalized=(features-self.head.feature_mean)/self.head.feature_std
        projected=torch.nn.functional.normalize(self.head.projection(normalized),dim=-1)
        # Center by the query, not by the mean of whatever candidate subset the
        # caller supplies. Candidate order/count therefore cannot shift scores.
        texts=torch.nn.functional.normalize(candidates-query,dim=-1)
        return projected@texts.T*self.head.log_scale.clamp(0,math.log(100)).exp()+self.head.bias

def normalized_visual(bundle, mean, std):
    x=torch.cat([bundle['visual'],bundle['temporal']],dim=-1)
    return torch.nn.functional.normalize((x-mean)/std,dim=-1)

def prototype_scores(bundle, obj, fitted, scale, candidate_states=None, fallback_scores=None):
    z=normalized_visual(bundle,fitted['mean'],fitted['std'])
    entry=fitted['by_object'].get(obj,fitted['global'])
    prior=entry['prior'].clamp(.001,.999)
    logits=torch.logit(prior)[None]+scale*(z@entry['positive'].T-z@entry['negative'].T)
    scores=logits.sigmoid().numpy()
    if candidate_states is None:return scores
    if not candidate_states or len(set(candidate_states))!=len(candidate_states):
        raise ValueError('Candidates must be nonempty and unique')
    columns={name:i for i,name in enumerate(fitted['states'])}
    output=[]
    for i,name in enumerate(candidate_states):
        if name in columns:output.append(scores[:,columns[name]])
        elif fallback_scores is not None and fallback_scores.shape==(len(scores),len(candidate_states)):
            output.append(fallback_scores[:,i])
        else:raise ValueError('Unseen candidate descriptions need text-scoring fallback')
    import numpy as np
    return np.stack(output,axis=1)
