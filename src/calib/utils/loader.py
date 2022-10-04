import pdb
import math
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
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
        self.num_level = len(opt.model_cfg.scales)
        self.intmats = data['Kmat'] # [N_patch, 3, 3] * level
        self.normals = data['normal']  # [N_patch, 3] * level
        self.cleans = data['sharp']  # [N_patch, 1, H, W] * level
        self.focus = data['focus']  # [N_patch, 1, H, W] * level
        self.blurs = data['blur']  # [N_patch, 1, H, W] * level
        self.depths = data['depth']  # [N_patch, 1, H, W] * level
        self.uv_coord = data['uv_coord']  # [N_patch, 1, H, W, 2] * level
        
        # Since the depth label of PSF is imbalanced, adapt "Delving into Deep Imbalanced Regression (ICML'21)" to regress imbalance
        # For LDS method, compute effective label density and re-weight
        self.weights_LDS, self.min_depth, self.max_depth = self.calc_reweight_LDS(data['depth'][0])
        
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
        scaling = np.sum(bins_number) / np.sum(np.array(bins_number) / (np.array(smoothed_bins) + 1e-6))
        weights = np.float32(scaling / (smoothed_bins + 1e-6)).clip(0, 1)
        return weights, depth_all.min(), depth_all.max()
        
    def __getitem__(self, index):
        
        sample_out = dict()
        type_det = np.float16 if self.opt.precision == 16 else np.float32
        for i in range(self.num_level):
            sample_out.update({'clean_{}'.format(i): [], 'blur_{}'.format(i): [], 'focus_{}'.format(i): [], 
                               'depth_{}'.format(i): [], 'normal_{}'.format(i): [], 'weight_{}'.format(i): [], 
                               'uv_coord_{}'.format(i): [], 'intmat_{}'.format(i): [], 'invKmat_{}'.format(i): []})
            
            # normal and intrinsics
            normal = type_det(self.normals[i][index])
            intmat = type_det(self.intmats[i][index])
            invKmat = type_det(np.linalg.inv(self.intmats[i][index]))
            normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = normal / normal_norm
            sample_out['normal_{}'.format(i)] = torch.tensor(normal[None])  # [1, 3]
            sample_out['intmat_{}'.format(i)] = torch.tensor(intmat[None])  # [1, 3, 3]
            sample_out['invKmat_{}'.format(i)] = torch.tensor(invKmat[None])  # [1, 3, 3]
            
            # preprocess data
            cleans = type_det(self.cleans[i][index] / 255.0)
            depths = type_det(self.depths[i][index])
            blurs = type_det(self.blurs[i][index])
            focus = type_det(self.focus[i][index])
            
            # Based on LDS, calc weight mask, to resolve imbalanced samples along depth
            if self.opt.use_LDS:
                ind = (depths - self.min_depth) / (self.max_depth - self.min_depth) * (int(self.opt.level / self.opt.LDS_step) - 1)
                ind_0 = np.int64(ind)
                ind_1 = np.clip(ind_0 + 1, 0, int(self.opt.level / self.opt.LDS_step) - 1)
                val_0 = self.weights_LDS[ind_0]
                val_1 = self.weights_LDS[ind_1]
                weight = type_det(val_0 * (ind_1 - ind) + val_1 * (ind - ind_0))  # linear interpolation
            else:
                weight = np.ones_like(depths)
            
            # convert to tensor
            # each tensor is in [X, Y, Z] coordinate
            sample_out['clean_{}'.format(i)] = torch.tensor(cleans)  # [1, H, W]
            sample_out['blur_{}'.format(i)] = torch.tensor(blurs)  # [1, H, W]
            sample_out['focus_{}'.format(i)] = torch.tensor(focus)  # [1, H, W]
            sample_out['depth_{}'.format(i)] = torch.tensor(depths)  # [1, H, W]
            sample_out['weight_{}'.format(i)] = torch.tensor(weight)  # [1, H, W]
            sample_out['uv_coord_{}'.format(i)] = torch.tensor(self.uv_coord[i][index].transpose(0, 3, 1, 2)[0])  # [1, 2, H, W]
        
        return sample_out
    
    def calib_collate_fn(self, batch):
        output = dict()
        for i in range(self.num_level):
            output_i = dict()
            for element in batch:
                for key in element.keys():
                    if '_{}'.format(i) not in key:
                        continue
                    rkey = key.replace('_{}'.format(i), '')
                    if rkey not in output_i.keys():
                        output_i[rkey] = []
                    output_i[rkey].append(element[key])
            for key in output_i.keys():
                if key not in output.keys():
                    output[key] = []
                output[key].append(torch.stack(output_i[key], dim=0))
        return output
    
    def __len__(self):
        return len(self.blurs[0])
    
    
