import math
import torch
import torch.nn.functional as F
from einops import rearrange


def masked_mean(tensor, mask, dim, keepdim=True):
    masked = torch.mul(tensor, mask)  # Apply the mask using an element-wise multiply
    return masked.sum(dim=dim, keepdim=keepdim) / (mask.sum(dim=dim, keepdim=keepdim) + 1e-7)  # Find the average!


def masked_minmax(tensor, mask, dim):
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    pos_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf  # Place the smallest values possible in masked positions'
    pos_inf[~mask] = math.inf  # Place the largest values possible in masked positions
    return (masked + pos_inf).min(dim=dim, keepdim=True)[0], (masked + neg_inf).max(dim=dim, keepdim=True)[0]
    
    
def normalize(img, mask=None, padvalue=0.0, eps=1e-6):
    batch, c, h, w = img.shape
    nimg = img.view(batch, c, -1)
    if mask is not None:
        nmask = mask.view(batch, c, -1)
        minvalue, maxvalue = masked_minmax(nimg, nmask, -1)
    else:
        minvalue = nimg.min(-1, keepdim=True)[0]
        maxvalue = nimg.max(-1, keepdim=True)[0]
    nimg = (nimg - minvalue) / (maxvalue - minvalue + eps)
    
    if mask is not None:
        return nimg.view(batch, c, h, w) * mask + padvalue * (~mask)
    else:
        return nimg.view(batch, c, h, w)
    
    
def unfolding(tensor, kernel_size):
    unfolded_tensor = F.unfold(tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, dilation=1)
    unfolded_tensor = rearrange(unfolded_tensor, 'b (c kh kw) l -> b l kh kw c', kh=kernel_size, kw=kernel_size)
    return unfolded_tensor