
import pdb
import torch
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
        
        # declare the model
        file = run_path('src/calib/model/%s/mainmodel.py' % opt.model)
        self.model = file['PSFVolume'](data['meta_data'], opt)
        self.board = board
        self.ckpt_name = None
    
    def train(self, patches, ckpt=None):
        
        # savepath
        rootpath = makedir_custom(Path('workspace'))  # rootpath
        rootpath = makedir_custom(rootpath / self.opt.calibname)  # workspace
        modelpath = makedir_custom(rootpath / self.opt.model)  # model workspace (save model related files here)
        ckptpath = makedir_custom(modelpath / 'ckpt')  # ckpt workspace (save checkpoint here)
        resultpath = makedir_custom(modelpath / 'result')  # save result here
        
        # Learning PSF from volume (ours)
        logger = pl_loggers.TensorBoardLogger(str(modelpath))
        lr_monitor = LearningRateMonitor(logging_interval='step')
        cp_monitor = ModelCheckpoint(
            dirpath=str(modelpath),
            filename=None,
            save_top_k=1, 
            save_last=True
        )
        callbacks = [lr_monitor, cp_monitor, TVLoss()] if self.opt.model_cfg.use_tv else [lr_monitor, cp_monitor]
        
        # dataloader setting
        pdb.set_trace()
        loader_ = DPCalloader(patches['patches'], True, self.opt)
        dataloader = DataLoader(loader_, batch_size=self.opt.batchsize,
                                shuffle=True, num_workers=self.opt.num_workers, 
                                in_memory=True)
        
        # for visualization
        vis_index = np.random.choice(len(loader_), opt.model_cfg.num_vis, replace=False).tolist()  # type: ignore
        
        # setting model detail
        if self.opt.model_cfg.multi_res:
            psfV_scales = self.opt.model_cfg.scales / self.opt.model_cfg.scales[0]
            milestones = self.opt.model_cfg.milestones
            callbacks.append(Volume_Scheduler(psfV_scales, milestones))
        cfg = {'vis_idx': vis_index, 'modelpath': modelpath, 'resultpath': resultpath}
        self.model.model_setting(cfg)
        
        # start training
        runner = Trainer(
            logger=logger,
            callbacks=callbacks,
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            strategy=self.opt.accelerator,
            benchmark=True,
            deterministic=False,
            gpus=self.opt.ngpus,
            precision=self.opt.precision,
            max_epochs=self.opt.epoch,
            sync_batchnorm=self.opt.sync_batch,
            num_sanity_val_steps=0,
            profiler="pytorch"
        )
        
        ckpt_path = str(ckptpath / ckpt) if ckpt is not None else None
        runner.fit(model=self.model, ckpt_path=ckpt_path)
        
        pass
    
    def test(self):
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





