
import os
from pytorch_lightning.callbacks import Callback


class TVLoss(Callback):
    def on_after_backward(self, trainer, pl_module):
        weight = pl_module.opt.model_cfg.loss_weights[3] * pl_module.psfV_scale
        pl_module.psf_volume.total_variation_add_grad(weight, weight, weight)