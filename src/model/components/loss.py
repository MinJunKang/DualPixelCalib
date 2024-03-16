import math
import torch
import numpy as np


def CharbonnierLoss_L1(x, scale=0.1):
    # alpha=1: Charbonnier/pseudo-Huber loss.
    assert torch.is_tensor(x)
    
    # This will be used repeatedly.
    squared_scaled_x = (x / scale) ** 2
    return torch.pow(squared_scaled_x + 1., 0.5) - 1.
    
    
def CharbonnierLoss_L2(x, scale=0.1):
    # alpha=2: L2 loss.
    assert torch.is_tensor(x)
    
    # This will be used repeatedly.
    squared_scaled_x = (x / scale) ** 2

    # The loss when alpha == 2.
    return 0.5 * squared_scaled_x


def SSIM(x, y, conf=None, window_size=11):
    """ Compute the structural similarity index between two images

    Args:
        x: (n_batch, n_dim, nx, ny) input image
        y: (n_batch, n_dim, nx, ny) input image
        conf: (n_batch, n_dim, nx, ny) input confidence map

    Returns:
        (float) structural similarity measure
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.nn.AvgPool2d(window_size, 1, padding=window_size//2)(x)
    mu_y = torch.nn.AvgPool2d(window_size, 1, padding=window_size//2)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = torch.nn.AvgPool2d(window_size, 1, padding=window_size//2)(x * x) - mu_x_sq
    sigma_y = torch.nn.AvgPool2d(window_size, 1, padding=window_size//2)(y * y) - mu_y_sq
    sigma_xy = torch.nn.AvgPool2d(window_size, 1, padding=window_size//2)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    if conf is not None:
        return torch.clamp((1 - SSIM) / 2, 0, 1) * torch.nn.AvgPool2d(window_size, 1, padding=window_size//2)(conf)
    else:
        return torch.clamp((1 - SSIM) / 2, 0, 1)
    
