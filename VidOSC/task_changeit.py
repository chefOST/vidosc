#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import pytorch_lightning as pl
import lookforthechange

from dataset_changeit import ChangeItFeatCLIPLabelDataset
from loader import construct_loader
from model import FeatTimeTransformerDualHead


class FrameCls(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ds = ChangeItFeatCLIPLabelDataset(args, "train")

        args.input_dim, self.n_classes, self.classid2name, args.state_classes, args.action_classes, args.object_classes, \
        self.state_id_mapping, self.action_id_mapping, self.action_to_state, args.vocab_size = self.train_ds.get_state_action_mapping()
        self.action_bg_idx = self.train_ds.action_bg_idx

        self.model = FeatTimeTransformerDualHead(args)

        self.state_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.action_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')


    def training_step(self, batch, batch_idx):
        input, state_pl, action_pl, video_level_score = batch   # text_embed 16, 768
        bs = input.shape[0]
        pred = self.model(input)  # [bs * ts, n_classes]

        state_loss_unweighted = self.state_loss(pred['state'], state_pl.view(-1))
        state_loss = (state_loss_unweighted.reshape(bs, -1) * video_level_score[:, None]).sum() / (state_pl != -1).sum()
        action_loss_unweighted = self.action_loss(pred['action'], action_pl.view(-1))
        action_loss = (action_loss_unweighted.reshape(bs, -1) * video_level_score[:, None]).sum() / (action_pl != -1).sum()
        loss = state_loss + action_loss

        self.log("train_state_loss", state_loss, on_epoch=True)
        self.log("train_action_loss", action_loss, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def model_inference(self, input):
        pred = self.model(input)
        st_probs = torch.softmax(pred['state'].unsqueeze(0), -1)  # (1, 900, 2*n_classes+1)
        ac_probs = torch.softmax(pred['action'].unsqueeze(0), -1)
        return st_probs, ac_probs


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        return construct_loader(self.args, "train", self.train_ds)


