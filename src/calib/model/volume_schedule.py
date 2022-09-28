
import numpy as np

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch_lightning.callbacks import Callback


class Volume_Scheduler(Callback):
    def __init__(self, opt):
        self.opt = opt
        self.psfV_scales = np.array(self.opt.model_cfg.scales)
        self.milestones = np.array(opt.model_cfg.milestones)
        assert(len(self.milestones) == len(self.psfV_scales) - 1)
        
    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        
        # find current stage
        diff = self.milestones - (trainer.current_epoch + 1)
        masked_diff = np.ma.array(diff, mask=diff <= 0)
        idx = masked_diff.argmin(fill_value=self.milestones.max()) if np.ma.count_masked(masked_diff) < len(self.milestones) else len(self.milestones)

        # schedule volume
        prev_stage = pl_module.stage
        current_stage = idx
        
        if prev_stage != current_stage:
            
            # update psf volume parameters
            pl_module.stage = current_stage
            pl_module.psfV_scale = self.psfV_scales[current_stage]
            newsize = (pl_module.levels[current_stage], pl_module.psfV_sizes[current_stage], pl_module.psfV_sizes[current_stage])
            pl_module.psf_volume.scale_volume_grid(newsize)
            
            # reset optimizer's parameters
            for param in optimizer.param_groups:
                param['params'].clear()
            optimizer.state.clear()
            optimizer.add_param_group({'params': pl_module.parameters()})
            
            print('Volume size changed from %f to %f !!' % (pl_module.psfV_sizes[prev_stage], pl_module.psfV_sizes[current_stage]))