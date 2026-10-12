#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import argparse
import os
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from task_changeit import FrameCls


def parse_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--dataset_root", type=str, default="./data_changeit")
    parser.add_argument("--ignore_video_weight", action="store_true")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--novel_obj", action="store_true") # ChangeIt open-world
    parser.add_argument("--bg_thre", default=0.5, type=float)
    parser.add_argument("--state_thre1", default=0.35, type=float)
    parser.add_argument("--state_thre2", default=0.37, type=float)
    parser.add_argument("--action_thre", default=0.4, type=float)
    parser.add_argument("--det", default=0, type=int, choices=[0, 1])
    # model args
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_dim", type=int, default=512)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    # training args
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")  # gpu / mps / cpu / auto
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--log_name", type=str, default="debug")
    parser.add_argument("--ckpt", type=str, default="")

    parser.add_argument("--use_gt_action", action="store_true")  # multitask use gt state transition idx
    parser.add_argument("--ordering", action="store_true") # use casual ordering during inference

    # weights & biases (disabled unless --wandb_project is set; bare --wandb_project logs to vidosc-experiments)
    parser.add_argument("--wandb_project", type=str, nargs="?", const="vidosc-experiments", default=None)
    parser.add_argument("--wandb_entity", type=str, default="chefOST")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)

    return parser.parse_args()


def main():
    log_dir = os.path.join(args.log_dir, 'changeit', args.log_name)
    os.makedirs(log_dir, exist_ok=True)
    print('Logging to:', log_dir)
    with open(f'{log_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    task = FrameCls(args)

    checkpoint_callbacks = []
    if hasattr(task, 'metric_name_list'):
        for name in task.metric_name_list:
            checkpoint_callbacks.append(
                ModelCheckpoint(
                    monitor=name,
                    filename=task.ckpt_save_format,
                    save_top_k=1,
                    mode='max',
                )
            )

    loggers = [TensorBoardLogger(save_dir=log_dir)]
    if args.wandb_project:
        wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                                   name=args.wandb_name or args.log_name, tags=args.wandb_tags,
                                   job_type="eval" if args.ckpt else "train", save_dir=log_dir)
        wandb_logger.experiment.config.update(vars(args), allow_val_change=True)
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator=args.accelerator,
        max_epochs=args.n_epochs,
        default_root_dir=log_dir,
        logger=loggers,
        callbacks=checkpoint_callbacks,
        num_sanity_val_steps=0,
    )

    if args.ckpt != "":
        task = FrameCls.load_from_checkpoint(checkpoint_path=args.ckpt, args=args)
        print(f'Evaluating {args.ckpt}')
        metrics = dict(trainer.validate(task)[0])
        if args.novel_obj:  # novel split
            metrics.update(trainer.test(task)[0])
        with open(f'{log_dir}/eval_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    else:
        trainer.fit(task)

    if args.wandb_project:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    main()
