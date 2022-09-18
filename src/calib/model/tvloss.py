
import os
from pytorch_lightning.callbacks import Callback


# total variance loss
from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)


class TVLoss(Callback):
    def on_after_backward(self, trainer, pl_module):
        if pl_module.psf_volume.grad is not None:
            # add total variation loss
            weight = pl_module.params['arg'].loss_weights[3] * pl_module.psfV_scale
            total_variation_cuda.total_variation_add_grad(pl_module.psf_volume, pl_module.psf_volume.grad, weight, weight, weight, True)  # type: ignore