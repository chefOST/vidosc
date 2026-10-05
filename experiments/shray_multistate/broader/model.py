"""Own query-conditioned text head; existing VidOSC source is imported read-only."""
import torch
from protocol import HERE
from temporal_states import TextStateHead
from prepare import load_upstream_encoder

class QueryStateModel(torch.nn.Module):
    def __init__(self,motion_dim=0,train_encoder=False):
        super().__init__()
        self.encoder=load_upstream_encoder()
        self.encoder.requires_grad_(train_encoder)
        self.train_encoder=train_encoder
        self.motion_dim=motion_dim
        self.head=TextStateHead(768+512+768+motion_dim,768,256)

    def context(self,visual):
        self.encoder.eval()  # Keep encoder dropout disabled during fine tuning.
        inp=torch.cat([visual,torch.zeros_like(visual)],dim=-1).unsqueeze(0)
        if self.train_encoder and self.training:return self.encoder(inp,middle=True).squeeze(0)
        with torch.no_grad():return self.encoder(inp,middle=True).squeeze(0)

    def features(self,visual,query,motion=None,temporal=None):
        if visual.ndim!=2 or visual.shape[1]!=768 or query.shape!=(768,):raise ValueError('Visual/query dimensions mismatch')
        if temporal is None:temporal=self.context(visual)
        values=[visual,temporal,query.expand(len(visual),-1)]
        if self.motion_dim:
            if motion is None or motion.shape!=(len(visual),self.motion_dim):raise ValueError('Motion features must align')
            values.append(motion)
        return torch.cat(values,dim=-1)

    def forward(self,visual,query,candidates,motion=None,temporal=None):
        return self.head(self.features(visual,query,motion,temporal),candidates)
