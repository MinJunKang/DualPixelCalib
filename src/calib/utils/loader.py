import pdb
import math
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms.functional as FT
from scipy.ndimage import convolve1d, gaussian_filter1d


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
        nmask = mask.view(batch, 1, -1)
        minvalue, maxvalue = masked_minmax(nimg, nmask, -1)
    else:
        minvalue = nimg.min(-1, keepdim=True)[0]
        maxvalue = nimg.max(-1, keepdim=True)[0]
    nimg = (nimg - minvalue) / (maxvalue - minvalue + eps)
    
    if mask is not None:
        return nimg.view(batch, c, h, w) * mask + padvalue * (~mask)
    else:
        return nimg.view(batch, c, h, w)
    
    
class DPCalloader(data.Dataset):
    
    def __init__(self, data, training, opt, repeat=False):
        
        self.opt = opt
        self.training = training
        self.repeat = repeat
        
        # collected data
        self.intmats = data['K']
        self.cleans = data['sharp']
        self.blurs = data['blur']
        self.depths = data['depth']
        self.normals = data['normal']
        self.uv_coord = data['uv_coord']
        pdb.set_trace()
        
        # Since the depth label of PSF is imbalanced, adapt "Delving into Deep Imbalanced Regression (ICML'21)" to regress imbalance
        # For LDS method, compute effective label density and re-weight
        depth = data[0]['DPc_patch']['depth']
        self.weights_LDS, self.min_depth, self.max_depth = self.calc_reweight_LDS(depth)
        
    def gauss1d_kernel_LDS(self):
        half_ks = (self.opt.LDS_ks - 1) // 2
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel = gaussian_filter1d(base_kernel, sigma=self.opt.LDS_sigma)
        return kernel / max(kernel)
        
    def calc_reweight_LDS(self, depths):
        depth_all = depths[depths > 0]
        bins_number, _ = np.histogram(depth_all, bins=int(self.opt.model_cfg.level / self.opt.LDS_step))
        bins_number = np.sqrt(bins_number)
        kernel_window = self.gauss1d_kernel_LDS()
        smoothed_bins = convolve1d(bins_number, weights=kernel_window, mode='reflect')
        scaling = np.sum(bins_number) / np.sum(np.array(bins_number) / np.array(smoothed_bins))
        weights = np.float32(scaling / smoothed_bins)
        return weights, depth_all.min(), depth_all.max()
        
    def __getitem__(self, index):
        
        # convert to PIL
        cleans = Image.fromarray(np.squeeze(self.cleans[index]) / 255.0)
        blurs = Image.fromarray(np.squeeze(self.blurs[index]))
        depths = Image.fromarray(np.squeeze(self.depths[index]))
        if self.repeat:
            normal = np.repeat(self.normals[index][None, :], depths.size[0] * depths.size[1], axis=0)
            normal = np.float32(normal.reshape(depths.size[1], depths.size[0], 3))
            intmat = np.repeat(self.intmats[index][None, :, :], depths.size[0] * depths.size[1], axis=0)
            intmat = np.float32(intmat.reshape(depths.size[1], depths.size[0], -1))
            magnitude = np.sqrt(np.sum(np.square(normal), axis=2))
            normal = normal / np.dstack((magnitude, magnitude, magnitude))
        else:
            normal = np.float32(self.normals[index][:, None])
            intmat = np.float32(self.intmats[index][:, :, None])
            magnitude = np.sqrt(np.sum(np.square(normal), axis=0))
            normal = normal / np.stack((magnitude, magnitude, magnitude))
        
        # convert to tensor
        cleant = FT.to_tensor(cleans)
        blurt = FT.to_tensor(blurs)
        deptht = FT.to_tensor(depths)
        uv_coordt = FT.to_tensor(np.squeeze(self.uv_coord[index]))
        normalt = FT.to_tensor(normal)
        intmatt = FT.to_tensor(intmat)
        
        # Based on LDS, calc weight mask, to resolve imbalanced samples along depth
        if self.opt.use_LDS:
            ind = (depths - self.min_depth) / (self.max_depth - self.min_depth) * (int(self.opt.level / self.opt.LDS_step) - 1)
            ind_0 = np.int64(ind)
            ind_1 = np.clip(ind_0 + 1, 0, int(self.opt.level / self.opt.LDS_step) - 1)
            val_0 = self.weights_LDS[ind_0]
            val_1 = self.weights_LDS[ind_1]
            weight = np.float32(val_0 * (ind_1 - ind) + val_1 * (ind - ind_0))  # linear interpolation
        else:
            weight = np.float32(np.ones_like(depths))
        weightt = FT.to_tensor(weight)
        
        sample_out = {'clean': cleant, 'blur': blurt, 'depth': deptht, 'normal': normalt, 'weight': weightt, 'uv_coord': uv_coordt, 'intmat': intmatt}
        
        return sample_out
    
    def __len__(self):
        return len(self.cleans)