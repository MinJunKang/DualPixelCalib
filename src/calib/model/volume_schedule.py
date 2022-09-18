
import numpy as np

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch_lightning.callbacks import Callback


class Volume_Scheduler(Callback):
    def __init__(self, psfV_scales, milestones):
        self.psfV_scales = psfV_scales
        self.milestones = np.array(milestones)
        
    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        
        # find current stage
        diff = self.milestones - (trainer.current_epoch + 1)
        masked_diff = np.ma.array(diff, mask=diff <= 0)
        idx = masked_diff.argmin(fill_value=self.milestones.max()) if np.ma.count_masked(masked_diff) < len(self.milestones) else -1
        
        # schedule volume
        prev_psfV_scale = pl_module.psfV_scale
        current_psfV_scale = self.psfV_scales[idx]
        
        if prev_psfV_scale != current_psfV_scale:
            
            # update psf volume parameters
            pl_module.psfV_scale = current_psfV_scale
            pl_module.level = int(pl_module.max_level * current_psfV_scale)
            pl_module.psfV_size = int(pl_module.max_psfV_size * current_psfV_scale)
            pl_module.psfV_size = (pl_module.psfV_size + 1) if pl_module.psfV_size % 2 == 0 else pl_module.psfV_size  # odd number
            pl_module.psf_volume = Parameter(F.interpolate(pl_module.psf_volume, size=(pl_module.level, pl_module.psfV_size, pl_module.psfV_size), mode='trilinear', align_corners=True), requires_grad=True)
            
            # reset optimizer's parameters
            for param in optimizer.param_groups:
                param['params'].clear()
            optimizer.state.clear()
            optimizer.add_param_group({'params': pl_module.parameters()})
            
            print('Volume scale changed from %f to %f !!' % (prev_psfV_scale, current_psfV_scale))