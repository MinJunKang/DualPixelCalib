

import lpips
import torch
import numpy as np
from texttable import Texttable
from .pohsun_ssim import pytorch_ssim


class CalibMetric(object):
    
    def __init__(self, logpath, samplenum=-1):
        self.index = 0
        self.t = Texttable()
        self.t.set_deco(Texttable.HEADER)
        self.samplenum = samplenum
        self.logpath = logpath
        
        self.metric = dict()
        self.metric['rmse'] = []
        self.metric['ssim'] = []
        self.metric['psnr'] = []
        self.metric['lpips'] = []
        self.lpips_loss = lpips.LPIPS(net="alex").cuda()
    
    def psnr(self, src, target, valid_mask=None):
        
        if valid_mask is not None:
            src *= valid_mask
            target *= valid_mask
        mse = ((src - target) ** 2)
        rmse = mse.sqrt().mean()
        psnr = -10 * mse.mean().log10()
        return np.float32(psnr.cpu().numpy()), np.float32(rmse.cpu().numpy())

    def ssim(self, src, target, valid_mask=None):
        
        if valid_mask is not None:
            src *= valid_mask
            target *= valid_mask
        return pytorch_ssim.ssim(src, target).item()
    
    def lpips(self, src, target, valid_mask=None):
        
        if valid_mask is not None:
            src *= valid_mask
            target *= valid_mask
        return self.lpips_loss(src, target).item()
    
    @torch.no_grad()
    def measure(self, results, log=True):
        
        pred = results['pred']
        target = results['target']
        mask = results['mask']
            
        # compute psnr, rmse
        psnr_value, rmse_value = self.psnr(pred, target, mask)
        
        # compute ssim
        ssim_value = self.ssim(pred, target, mask)
        
        # compute lpips
        lpips_value = self.lpips(pred, target, mask)
        
        # log 
        if log:
            data = {'rmse': rmse_value, 'ssim': ssim_value, 'psnr': psnr_value, 'lpips': lpips_value}
            self.update(data)
            
        return data
        
    def update(self, data):
    
        if (self.samplenum != -1) and (self.index >= self.samplenum):
            return

        for key in self.metric.keys():
            self.metric[key].append(data[key])

        self.index += 1
        
    def get_value(self, pos=-1, use_chart=False):
    
        results = []

        if self.index == 0:
            return None, None

        if pos == -1:
            for key in self.metric.keys():
                results.append(np.array(self.metric[key]).mean())
        else:
            for key in self.metric.keys():
                results.append(self.metric[key][pos])

        if use_chart:
            t = self.t
            t.set_deco(Texttable.HEADER)
            t.set_cols_dtype(['f' for key in self.metric.keys()])
            t.set_cols_width([10 for key in self.metric.keys()])
            t.add_row([key for key in self.metric.keys()])
            t.add_rows([results], header=False)
            t.set_cols_align(['r' for key in self.metric.keys()])
            
            # log as text
            with open(str(self.logpath / "metrics.txt"), "w") as f:
                print(t.draw(), file=f)
            
            return results, t
        else:
            return results
        
    def get_value_test(self, pos=-1, use_chart=False):
        
        results = []
        results_var = []

        if self.index == 0:
            return None, None

        if pos == -1:
            for key in self.metric.keys():
                mat = np.array(self.metric[key])
                results.append(mat.mean())
                results_var.append((mat ** 2).mean() - (mat.mean() ** 2))  # Var[x] = E[x^2] - E[x]^2
        else:
            for key in self.metric.keys():
                results.append(self.metric[key][pos])

        if use_chart:
            t = self.t
            t.set_deco(Texttable.HEADER)
            t.set_cols_dtype(['f' for key in self.metric.keys()])
            t.set_cols_width([10 for key in self.metric.keys()])
            t.add_row([key for key in self.metric.keys()])
            t.add_rows([results], header=False)
            t.add_rows([results_var], header=False)
            t.set_cols_align(['r' for key in self.metric.keys()])
            
            # log as text
            with open(str(self.logpath / "metrics.txt"), "w") as f:
                print(t.draw(), file=f)
            
            return results, t
        else:
            return results
        

    def clear(self):
        self.index = 0
        for key in self.metric.keys():
            self.metric[key] = []