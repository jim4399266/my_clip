import os
import copy
import torch
import pytorch_lightning as pl
from pathlib import Path
import os
# from torchstat import stat
import pytorch_lightning.loggers
import random
os.environ["NCCL_DEBUG"] = "INFO"

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

from config import ex
# from datamodules.multitask_datamodule import MTDataModule
from modules.model_module import ModelModule
from datamodules import build_datamodule

@ex.automain
def main(_config):
    config = copy.deepcopy(_config)
    del _config

    # 如果不用GPU，则num_gpus=0，防止下面除0，num_gpus置为1
    num_gpus = config['gpus'] if isinstance(config['gpus'], int) else len(config['gpus'])
    config['num_gpus'] = num_gpus
    config['dist'] = True if num_gpus > 1 else False
    strategy = 'ddp' if num_gpus > 1 else None
    config['grad_steps'] = max(config['batch_size'] // (
            config['per_gpu_batchsize'] * max(1, num_gpus) * config['num_nodes']
    ), 1)

    # TODO 添加动量更新策略
    dm = build_datamodule(config)
    model = ModelModule(config)

    log_dir = config['log_dir']
    output_dir = config['output_dir']

    if output_dir != None or "" or '':
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if config['load_path'] == "":
        m_p = 'momentum_'
        log_name = f'{m_p}bs{config["batch_size"]}_pbs{config["per_gpu_batchsize"]}_qs{config["queue_size"]}_epoch{config["max_epoch"]}_lr{config["learning_rate"]}_is{config["image_size"]}_from_{config["vit_name"]}_{config["image_size"]}_{config["tokenizer_name"]}'
        saved_dir = Path(output_dir) / f'{m_p}bs{config["batch_size"]}_pbs{config["per_gpu_batchsize"]}_qs{config["queue_size"]}_epoch{config["max_epoch"]}_lr{config["learning_rate"]}_from_{config["vit_name"]}_{config["image_size"]}_{config["tokenizer_name"]}'
    else:
        log_name = f'bs{config["batch_size"]}_pbs{config["per_gpu_batchsize"]}_qs{config["queue_size"]}_epoch{config["max_epoch"]}_lr{config["learning_rate"]}_is{config["image_size"]}_from_{config["load_path"].split("/")[-1][:-5]}'
        saved_dir = Path(output_dir) / f'bs{config["batch_size"]}_pbs{config["per_gpu_batchsize"]}_qs{config["queue_size"]}_epoch{config["max_epoch"]}_lr{config["learning_rate"]}_is{config["image_size"]}_from_{config["load_path"].split("/")[-1][:-5]}'

    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=log_name,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=saved_dir / f"version_{logger.version}" if output_dir != (None or "" or '') else None,
        filename='step{step}-val_score{val/the_metric:.4f}-val_loss{val/total_loss:.4f}',
        auto_insert_metric_name=False,
        save_top_k=3,
        monitor='val/the_metric',
        mode='max',
        save_last=False,
        verbose=True,
        save_weights_only=True,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    # earlystop_callback = pl.callbacks.EarlyStopping(monitor='val/total_loss', mode='min', check_on_train_epoch_end=False)
    # callbacks = [checkpoint_callback, lr_callback, earlystop_callback]
    callbacks = [checkpoint_callback, lr_callback]

    trainer = pl.Trainer(
        resume_from_checkpoint=config['load_path'],
        strategy=strategy,
        replace_sampler_ddp=False,
        logger=logger,
        log_every_n_steps=10,

        amp_backend='apex' if config['apex'] else "native",
        amp_level=config['amp_level'] if config['apex'] else None,
        precision=16 if config['apex'] else config['precision'],
        gpus=config['gpus'],

        # benchmark=True,
        max_epochs=config['max_epoch'],
        callbacks=callbacks,
        # gradient_clip_val=None if config['manual_optimization'] else config['max_grad_norm'],
        # accumulate_grad_batches=None if config['manual_optimization'] else grad_steps,
        gradient_clip_val=config['max_grad_norm'],
        accumulate_grad_batches=config['grad_steps'],
        weights_summary='top',
        fast_dev_run=config['fast_dev_run'],
        num_sanity_val_steps=config['num_sanity_val_steps'],
        val_check_interval=config['val_check_interval'],
    )

    if not config['test_only']:
        trainer.fit(model, datamodule=dm)

        weight_paths = list(Path(checkpoint_callback.dirpath).rglob('*.[pc][tk][hp]*'))
        # weight_paths = list(Path('/home/tzj/codes/my_clip/outputs/bs4096_pbs128_epoch20_lr8e-05_from_ViT-B-32_224_roberta-base/version_0').rglob('*.[pc][tk][hp]*'))
        for ckpt in weight_paths:
            trainer.test(model, datamodule=dm, ckpt_path=str(ckpt))

    else:
        trainer.test(model, datamodule=dm)
        # trainer.predict()
