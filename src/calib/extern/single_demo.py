
import pdb
import sys
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from einops.einops import rearrange
from pathlib import Path
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict


class Run_IFAN44(object):
    
    def __init__(self, rootpath):
        sys.path.append(rootpath)
        load_file = run_path(str(Path(rootpath) / "run_IFAN44.py"))
        self.config, self.model = load_file['IFAN44_create'](rootpath)
        self.img_multiple_of = self.config.refine_val
        
    def refine_image(self, img, val = 16):
        shape = img.shape
        if len(shape) == 4:
            _, _, h, w = shape[:]
            return img[:, :, 0 : h - h % val, 0 : w - w % val]
        elif len(shape) == 3:
            _, h, w = shape[:]
            return img[:, 0 : h - h % val, 0 : w - w % val]
        elif len(shape) == 2:
            h, w = shape[:]
            return img[0 : h - h % val, 0 : w - w % val]
        
    def padding(self, input_):
        
        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+self.img_multiple_of)//self.img_multiple_of)*self.img_multiple_of, ((w+self.img_multiple_of)//self.img_multiple_of)*self.img_multiple_of
        padh = H-h if h%self.img_multiple_of!=0 else 0
        padw = W-w if w%self.img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        
        return input_
        
    def forward(self, img_):
        
        with torch.no_grad():
            img = Image.fromarray(img_).convert('RGB')
            input_ = TF.to_tensor(img).unsqueeze(0).cuda()
            
            # if image is too big, split into parts and run independently : reduce memory usage
            if max(input_.shape[2], input_.shape[3]) > 3000:
                # split into 16 patches (4 by 4)
                h_, w_ = input_.shape[2] // 4, input_.shape[3] // 4
            elif max(input_.shape[2], input_.shape[3]) > 1500:
                # split into 4 patches (2 by 2)
                h_, w_ = input_.shape[2] // 2, input_.shape[3] // 2
            else:
                # just run whole image
                h_, w_ = input_.shape[2], input_.shape[3]
            
            # split images into patches
            images = F.unfold(input_, kernel_size=(h_, w_), stride=(h_, w_))
            images = rearrange(images, 'n (c h w) l -> n c h w l', h=h_, w=w_)
        
            outputs = []
            num_image = images.shape[-1]
            for i in range(num_image):
                input = self.padding(images[:, :, :, :, i])
                out = self.model(C=input, is_train=False)
                output = out['result']
                output = torch.clamp(output, 0, 1)
                # Unpad the output
                outputs.append(output[:,:,:h_,:w_])
            outputs = torch.stack(outputs, dim=-1)
            
            # stitch into a single image
            restored = F.fold(outputs.view(1, -1, num_image), 
                            output_size=(input_.shape[2], input_.shape[3]), 
                            kernel_size=(h_, w_), stride=(h_, w_))

            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])
        
        return restored


class Run_MPRNet(object):
    
    def __init__(self, rootpath, task):
        
        load_file = run_path(str(Path(rootpath) / task / "MPRNet.py"))
        self.model = load_file['MPRNet']()
        self.img_multiple_of = 8
        
        weights = str(Path(rootpath) / task / "pretrained_models" / ("model_"+task.lower()+".pth"))
        self.load_checkpoint(weights)
        self.model.cuda()
        self.model.eval()
        
    def load_checkpoint(self, weights):
        checkpoint = torch.load(weights)
        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            
    def padding(self, input_):
        
        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+self.img_multiple_of)//self.img_multiple_of)*self.img_multiple_of, ((w+self.img_multiple_of)//self.img_multiple_of)*self.img_multiple_of
        padh = H-h if h%self.img_multiple_of!=0 else 0
        padw = W-w if w%self.img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        
        return input_
            
    def forward(self, img_):
        
        with torch.no_grad():
            img = Image.fromarray(img_).convert('RGB')
            input_ = TF.to_tensor(img).unsqueeze(0).cuda()
            
            # if image is too big, split into parts and run independently : reduce memory usage
            if max(input_.shape[2], input_.shape[3]) > 3000:
                # split into 16 patches (4 by 4)
                h_, w_ = input_.shape[2] // 4, input_.shape[3] // 4
            elif max(input_.shape[2], input_.shape[3]) > 1500:
                # split into 4 patches (2 by 2)
                h_, w_ = input_.shape[2] // 2, input_.shape[3] // 2
            else:
                # just run whole image
                h_, w_ = input_.shape[2], input_.shape[3]
            
            # split images into patches
            images = F.unfold(input_, kernel_size=(h_, w_), stride=(h_, w_))
            images = rearrange(images, 'n (c h w) l -> n c h w l', h=h_, w=w_)

            outputs = []
            num_image = images.shape[-1]
            for i in range(num_image):
                input = self.padding(images[:, :, :, :, i])
                output = self.model(input)[0]
                output = torch.clamp(output, 0, 1)
                # Unpad the output
                outputs.append(output[:,:,:h_,:w_])
            outputs = torch.stack(outputs, dim=-1)
                    
            # stitch into a single image
            restored = F.fold(outputs.view(1, -1, num_image), 
                            output_size=(input_.shape[2], input_.shape[3]), 
                            kernel_size=(h_, w_), stride=(h_, w_))

            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])
        
        return restored
    
    
class Run_DeepRFT(object):
    
    def __init__(self, rootpath):
        
        self.div = 4
        self.win = 512
        self.num_res = 8
        self.img_multiple_of = 8
        
        load_file = run_path(str(Path(rootpath) / "DeepRFT_MIMO.py"))
        self.model = load_file['DeepRFT'](num_res=self.num_res, inference=True)
        
        weights = str(Path(rootpath) / 'DeepRFT/DeepRFT/model_DPDD.pth')
        self.load_checkpoint(weights)
        self.model.cuda()
        self.model.eval()
        
    def load_checkpoint(self, weights):
        checkpoint = torch.load(weights)
        old_state_dict = checkpoint["state_dict"]
        state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            # print(k)
            name = k
            if k[:7] == 'module.':
                name = k[7:]  # remove `module.`
            state_dict[name] = v
        # state_dict = checkpoint["state_dict"]
        do_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[-1] == 'W' and k[:-1] + 'D' in state_dict:
                k_D = k[:-1] + 'D'
                k_D_diag = k_D + '_diag'
                W = v
                D = state_dict[k_D]
                D_diag = state_dict[k_D_diag]
                D = D + D_diag
                # W = torch.reshape(W, (out_channels, in_channels, D_mul))
                out_channels, in_channels, MN = W.shape
                M = int(np.sqrt(MN))
                DoW_shape = (out_channels, in_channels, M, M)
                DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
                do_state_dict[k] = DoW
            elif k[-1] == 'D' or k[-6:] == 'D_diag':
                continue
            elif k[-1] == 'W':
                out_channels, in_channels, MN = v.shape
                M = int(np.sqrt(MN))
                W_shape = (out_channels, in_channels, M, M)
                do_state_dict[k] = torch.reshape(v, W_shape)
            else:
                do_state_dict[k] = v
        self.model.load_state_dict(do_state_dict)
            
    def window_partitions(self, x, window_size):
        """
        Args:
            x: (B, C, H, W)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, C, window_size, window_size)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
        return windows

    def window_reverses(self, windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, C, window_size, window_size)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, C, H, W)
        """
        C = windows.shape[1]
        x = windows.view(-1, H // window_size, W // window_size, C, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
        return x

    def window_partitionx(self, x, window_size):
        _, _, H, W = x.shape
        h, w = window_size * (H // window_size), window_size * (W // window_size)
        x_main = self.window_partitions(x[:, :, :h, :w], window_size)
        b_main = x_main.shape[0]
        if h == H and w == W:
            return x_main, [b_main]
        if h != H and w != W:
            x_r = self.window_partitions(x[:, :, :h, -window_size:], window_size)
            b_r = x_r.shape[0] + b_main
            x_d = self.window_partitions(x[:, :, -window_size:, :w], window_size)
            b_d = x_d.shape[0] + b_r
            x_dd = x[:, :, -window_size:, -window_size:]
            b_dd = x_dd.shape[0] + b_d
            # batch_list = [b_main, b_r, b_d, b_dd]
            return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
        if h == H and w != W:
            x_r = self.window_partitions(x[:, :, :h, -window_size:], window_size)
            b_r = x_r.shape[0] + b_main
            return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
        if h != H and w == W:
            x_d = self.window_partitions(x[:, :, -window_size:, :w], window_size)
            b_d = x_d.shape[0] + b_main
            return torch.cat([x_main, x_d], dim=0), [b_main, b_d]

    def window_reversex(self, windows, window_size, H, W, batch_list):
        h, w = window_size * (H // window_size), window_size * (W // window_size)
        x_main = self.window_reverses(windows[:batch_list[0], ...], window_size, h, w)
        B, C, _, _ = x_main.shape
        # print('windows: ', windows.shape)
        # print('batch_list: ', batch_list)
        res = torch.zeros([B, C, H, W],device=windows.device)
        res[:, :, :h, :w] = x_main
        if h == H and w == W:
            return res
        if h != H and w != W and len(batch_list) == 4:
            x_dd = self.window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
            res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
            x_r = self.window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
            res[:, :, :h, w:] = x_r[:, :, :, w - W:]
            x_d = self.window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
            res[:, :, h:, :w] = x_d[:, :, h - H:, :]
            return res
        if w != W and len(batch_list) == 2:
            x_r = self.window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
            res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        if h != H and len(batch_list) == 2:
            x_d = self.window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
            res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
            
    def forward(self, img_):
        
        with torch.no_grad():
            img = Image.fromarray(img_).convert('RGB')
            input_ = TF.to_tensor(img).unsqueeze(0).cuda()
            
            # partion image into windows
            _, _, Hx, Wx = input_.shape
            if min(Hx, Wx) < self.win:
                return img_
            win_size = self.win if min(Hx, Wx) >= self.win else 256
            
            input_re, batch_list = self.window_partitionx(input_, win_size)
            
            step = input_re.size(0) // self.div
            outputs = []
            for t in range(self.div):
                s = t * step
                if t == self.div-1:
                    div_input = input_re[s:]
                else:
                    div_input = input_re[s:s+step]
                
                outputs.append(self.model(div_input))
            restored = torch.cat(outputs, dim=0)
            restored = self.window_reversex(restored, win_size, Hx, Wx, batch_list)

            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])
        
            return restored
        