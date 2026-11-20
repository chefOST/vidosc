#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import lookforthechange

from dataset import build_vocab
from loader import construct_loader
from model import FeatTimeTransformer


class FrameCls(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.infer_ordering = False  # use causual ordering constraint during inference
        self.vocab, self.sc_list, _ = build_vocab(args)
        self.category_num = len(self.vocab) - 1
        args.vocab_size = 3 * self.category_num + 1
        args.input_dim = 768 * (1 + self.args.det)
        self.model = FeatTimeTransformer(args)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def training_step(self, batch, batch_idx):
        feat, pl = batch
        pred = self.model(feat)
        loss = self.loss(pred, pl.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def infer_state_idx(self, prob):
        pred_idx = torch.argmax(prob, dim=0).cpu().numpy()
        if self.infer_ordering:
            st_probs = torch.stack((prob[:, 1], prob[:, 3]), dim=-1).unsqueeze(0)
            ac_probs = prob[:, 2].unsqueeze(0).unsqueeze(2)
            # print('before', pred_idx)
            s0_idx, s2_idx, s1_idx = lookforthechange.optimal_state_change_indices(st_probs, ac_probs,
                lens=torch.tensor([st_probs.shape[1]], dtype=torch.int32, device=st_probs.device))[0].cpu().numpy()
            pred_idx = np.array([s0_idx, s1_idx, s2_idx])
            # print('after', pred_idx)
        else:
            pred_idx = pred_idx[1:]
        return pred_idx


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        return construct_loader(self.args, "train")

