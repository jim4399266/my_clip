import os
import copy
import torch
import argparse
import ruamel.yaml as yaml

import pytorch_lightning.loggers
import pytorch_lightning as pl
from pathlib import Path

# from datamodules.multitask_datamodule import MTDataModule
# from modules.model_module import ModelModule
from modules.blip_module import BLIPModule
from datamodules import build_datamodule

os.environ["NCCL_DEBUG"] = "INFO"
torch.set_float32_matmul_precision('high')

def main(args, config):
    # 如果不用GPU，则num_gpus=0，防止下面除0，num_gpus置为1
    config['num_device'] = config['devices'] if isinstance(config['devices'], int) else len(config['devices'])
    config['dist'] = True if config['num_device'] > 1 else False
    strategy = 'ddp' if config['num_device'] > 1 else None
    grad_steps = max(config['batch_size'] // (
            config['per_gpu_batchsize'] * max(1, config['num_device']) * config['num_nodes']
    ), 1)

    dm = build_datamodule(config)
    model = BLIPModule(config)

    log_dir = config['log_dir']
    if config['pretrained'] == "":
        task = config['task_name'].keys()
        log_name = f'{task}_bs{config["batch_size"]}_pbs{config["per_gpu_batchsize"]}_epoch{config["max_epoch"]}_lr{config["learning_rate"]}_from_{config["vit_name"]}_{config["image_size"]}_{config["tokenizer_name"]}'
    else:
        task = config['task_name'].keys()
        log_name = f'{task}_bs{config["batch_size"]}_pbs{config["per_gpu_batchsize"]}_epoch{config["max_epoch"]}_lr{config["learning_rate"]}_is{config["image_size"]}_from_{config["pretrained"].split("/")[-1].split(".")[0]}'
    output_dir = config['output_dir']
    if output_dir != None or "" or '':
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_dir = Path(output_dir) / log_name

    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=log_name,
        # default_hp_metric=False,    # 禁用 PyTorch Lightning 默认的 hparams 评估指标, 启用 TensorboardX
    )

    modelsummary_callback = pl.callbacks.ModelSummary(max_depth=1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=saved_dir / f"version_{logger.version}" if output_dir != (None or "" or '') else None,
        filename='step{step}-val_score{irtr/val/r_mean:.4f}',
        auto_insert_metric_name=False,
        save_top_k=3,
        monitor='irtr/val/r_mean',
        mode='max',
        save_last=False,
        verbose=True,
        save_weights_only=True,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    # earlystop_callback = pl.callbacks.EarlyStopping(monitor='val/total_loss', mode='min', check_on_train_epoch_end=False)
    # callbacks = [checkpoint_callback, lr_callback, earlystop_callback]
    callbacks = [modelsummary_callback, checkpoint_callback, lr_callback]

    trainer = pl.Trainer(
        # resume_from_checkpoint=config['load_path'],
        logger=logger,
        log_every_n_steps=10,
        precision=16 if config['apex'] else config['precision'],
        # amp_backend='apex' if config['apex'] else "native",
        # amp_level=config['amp_level'] if config['apex'] else None,

        accelerator=config['accelerator'],
        devices=config['devices'],
        # gpus=config['gpus'],
        strategy=strategy,
        use_distributed_sampler=False,

        # benchmark=True,
        max_epochs=config['max_epoch'],
        callbacks=callbacks,
        # gradient_clip_val=None if config['manual_optimization'] else config['max_grad_norm'],
        # accumulate_grad_batches=None if config['manual_optimization'] else grad_steps,
        gradient_clip_val=config['max_grad_norm'],
        accumulate_grad_batches=grad_steps,
        # weights_summary='top',
        fast_dev_run=config['fast_dev_run'],
        num_sanity_val_steps=config['num_sanity_val_steps'],
        val_check_interval=config['val_check_interval'],
    )

    if args.test_only:
        trainer.test(model, datamodule=dm)
        # trainer.predict()
    elif args.evaluate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)
        weight_paths = list(Path(checkpoint_callback.dirpath).rglob('*.[pc][tk][hp]*'))
        # weight_paths = list(Path('/home/tzj/codes/my_clip/outputs/bs4096_pbs128_epoch20_lr8e-05_from_ViT-B-32_224_roberta-base/version_0').rglob('*.[pc][tk][hp]*'))
        for ckpt in weight_paths:
            trainer.test(model, datamodule=dm, ckpt_path=str(ckpt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_coco.yaml')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if args.debug:
        config['fast_dev_run'] = 2
    main(args, config)