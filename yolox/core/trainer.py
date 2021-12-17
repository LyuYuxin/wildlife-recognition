#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0
    
        #tag lyx
        self.fix_lr = False
        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            os.path.join(self.file_name,self.timestamp),
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        # try:
        self.train_in_epoch()
        # except Exception:
        #     print(1)
        # finally:
        self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)
        
        if not self.fix_lr:
            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        else: 
            # lr = 2e-4 *  ((self.max_epoch - self.epoch - 1) / self.max_epoch) ** 2
            lr = 2.5e-4
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        self.model = self.exp.get_model()
        # logger.info(
        #     "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        # )
        self.model.to(self.device)

        # value of epoch will be set in `resume_train`
        self.model = self.resume_train(self.model)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        #tag lyx
        if self.args.finetune:
            #freeze backbone
            # for param in model.backbone.parameters():
            #     param.requires_grad = False
            
            # # freeze stem
            # for param in model.head.stems.parameters():
            #     param.requires_grad = False

            # freeze reg 
            # for param in model.head.reg_convs.parameters():
            #     param.requires_grad = False
            # for param in model.head.reg_preds.parameters():
            #     param.requires_grad = False

            # freeze cls
            # for param in model.head.cls_convs.parameters():
            #     param.requires_grad = False
            # for param in model.head.cls_preds.parameters():
            #     param.requires_grad = False
            
            #freeze obj
            # for param in model.head.obj_preds.parameters():
            #     param.requires_grad = False
            pass
    
        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)
       
        self.model.train()
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(self.model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )


        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(os.path.join(self.file_name, self.timestamp) )

        logger.info("Training start...")
        # logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            # if not self.no_aug:
            #     self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            #tag
            # gpu0 gg here.
            if self.rank == 0:
                #log loss to tensorboard
                for k, v in loss_meter.items():
                    self.tblogger.add_scalar("loss/" + k, v.latest, self.progress_in_iter + 1)
                
                #log lr to tensorboard
                self.tblogger.add_scalar("lr", self.meter['lr'].latest, self.progress_in_iter + 1)
            
            self.meter.clear_meters()
        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)

            #remove head
            # for key in list(ckpt['model'].keys()):
            #     if key.startswith("head.cls"):
            #         del ckpt['model'][key]      

            # resume the model/optimizer state dict
            if self.is_distributed:
                keys = [key[7:] for key in  ckpt['model'].keys()]
            
                ckpt["model"] = dict(zip(keys, ckpt['model'].values()))

            model.load_state_dict(ckpt["model"], strict=False)

            # self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                
                #remove head
                # for key in list(ckpt.keys()):
                #     if key.startswith("head.cls_preds"):
                #         del ckpt[key]            

                model = load_ckpt(model, ckpt)

            self.start_epoch = 0

        # self.fix_lr = True
        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        stats, summary = self.exp.eval(

            evalmodel, self.evaluator, self.is_distributed
        )
        (mAP, AP50, AP75, mAP_small, mAP_mid, mAP_large,\
            _, _, mAR, mAR_small, mAR_mid, mAR_large) = stats

        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/mAP", mAP, self.epoch + 1)
            self.tblogger.add_scalar("val/AP50", AP50, self.epoch + 1)
            self.tblogger.add_scalar("val/AP75", AP75, self.epoch + 1)
            self.tblogger.add_scalar("val/mAP_small", mAP_small, self.epoch + 1)
            self.tblogger.add_scalar("val/mAP_mid", mAP_mid, self.epoch + 1)
            self.tblogger.add_scalar("val/mAP_large", mAP_large, self.epoch + 1)

            self.tblogger.add_scalar("val/mAR", mAR, self.epoch + 1)
            self.tblogger.add_scalar("val/mAR_small", mAR_small, self.epoch + 1)
            self.tblogger.add_scalar("val/mAR_mid", mAR_mid, self.epoch + 1)
            self.tblogger.add_scalar("val/mAR_large", mAR_large, self.epoch + 1)

            logger.info("\n" + summary)
        synchronize()
        
        #tag lyx
        # self.save_ckpt(f"finetune_ibloss_epoch{self.epoch - self.start_epoch}_{ap50}", ap50_95 > self.best_ap)
        # self.save_ckpt(f"finetune_eqloss_epoch{self.epoch - self.start_epoch}_{ap50}", ap50_95 > self.best_ap)
        self.save_ckpt(f"epoch{self.epoch}_ap{AP50}", mAP > self.best_ap)
        self.best_ap = max(self.best_ap, mAP)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name + ckpt_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
