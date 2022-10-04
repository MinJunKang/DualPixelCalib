
import pdb
import torch
import numpy as np
from pathlib import Path
from runpy import run_path
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from src.calib.model.tvloss import TVLoss
from src.calib.utils.loader import DPCalloader
from src.calib.model.masked_adam import MaskedAdam
from src.calib.model.volume_schedule import Volume_Scheduler
from src.calib.utils.file_manager import makedir_custom


class PSFLearner(object):
    
    def __init__(self, data, opt, board):
        self.opt = opt
        
        # model paths
        self.metadata = data['meta_data']
        self.file = run_path('src/calib/model/%s/mainmodel.py' % opt.model)
        self.board = board
        self.ckpt_name = None
    
    def train(self, patches, raw, type, ckpt=None):
        '''
            train PSF volume of left, right, center individually
        '''
        
        # savepath
        cfg_name = self.opt.config.replace('config', '')
        rootpath = makedir_custom(Path('workspace'))  # rootpath
        rootpath = makedir_custom(rootpath / self.opt.calibname)  # workspace
        rootpath = makedir_custom(rootpath / ('%s_%s' % (self.board.tag, type)))  # workspace
        modelpath = makedir_custom(rootpath / (self.opt.model + cfg_name), True)  # model workspace (save model related files here)
        ckptpath = makedir_custom(modelpath / 'ckpt')  # ckpt workspace (save checkpoint here)
        resultpath = makedir_custom(modelpath / 'result')  # save result here
        
        # Learning PSF from volume (ours)
        logger = pl_loggers.TensorBoardLogger(str(modelpath), name='logs')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        cp_monitor = ModelCheckpoint(
            dirpath=str(ckptpath),
            filename=None,
            save_top_k=1, 
            save_last=True
        )
        callbacks = [lr_monitor, cp_monitor, TVLoss()] if self.opt.model_cfg.use_tv else [lr_monitor, cp_monitor]
        
        # dataloader setting
        loader_ = DPCalloader(patches['%s_patch' % type], True, self.opt)
        dataloader = DataLoader(loader_, batch_size=self.opt.batch_size,
                                shuffle=True, num_workers=self.opt.num_workers, 
                                collate_fn=loader_.calib_collate_fn, pin_memory=True)
        val_dataloader = DataLoader(loader_, batch_size=1,
                                    shuffle=False, num_workers=2, 
                                    collate_fn=loader_.calib_collate_fn, pin_memory=True)
        
        # declare the model
        self.model = self.file['PSFVolume'](self.metadata, self.opt)
        
        # psfV_scale
        psf_prj_scale = [max(scale) for scale in patches['%s_patch' % type]['scale']]
        
        # for visualization
        vis_index = np.random.choice(len(loader_), self.opt.model_cfg.num_vis, replace=False).tolist()  # type: ignore
        
        # setting model detail
        if len(self.opt.model_cfg.scales) > 1: callbacks.append(Volume_Scheduler(self.opt))
        cfg = {'psf_prj_scale': psf_prj_scale, 'img_size': (raw['h_dev'], raw['w_dev']), 
               'vis_idx': vis_index, 'modelpath': modelpath, 'resultpath': resultpath}
        self.model.model_setting(cfg)
        
        # start training
        runner = Trainer(
            logger=logger,
            callbacks=callbacks,
            enable_checkpointing=True,
            strategy=self.opt.accelerator,
            benchmark=True,
            deterministic=False,
            gpus=self.opt.ngpus,
            precision=self.opt.precision,
            max_epochs=self.opt.epoch,
            sync_batchnorm=self.opt.sync_batch,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=self.opt.model_cfg.record_epoch,
            profiler="pytorch",
            amp_backend='native'
        )
        
        # train begin
        ckpt_path = str(ckptpath / ckpt) if ckpt is not None else None
        runner.fit(model=self.model, train_dataloaders=dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)
        
    def train_all(self, patches, ckpt=None):
        '''
            train PSF volume of left, right, center all together
        '''
        pass
    
    def test(self):
        pass
    
    def test_all(self):
        pass


def optimizer_selector(params, option):
    
    if option.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=option.init_lr, betas=(0.9, 0.999), eps=1e-8)
    elif option.optim == 'madam':
        optimizer = MaskedAdam(params, lr=option.init_lr, betas=(0.9, 0.999), eps=1e-8)
    elif option.optim == 'radam':
        optimizer = torch.optim.RAdam(params, lr=option.init_lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    elif option.optim == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=option.init_lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8, amsgrad=False)
    elif option.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=option.init_lr, momentum=0.9, weight_decay=0.01)
    elif option.optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=option.init_lr, eps=1e-8)
    else:
        raise NotImplementedError('optimizer is not defined, please check your optimizer configuration !')

    return optimizer





