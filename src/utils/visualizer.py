
import cv2
import copy, time
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.utils.base import create_dir
import plotly.graph_objects as go

import dash
import parmap
from dash import dcc, html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def visualize_corner(img, corners, index, name='tmp.png', sizet=4):
    
    img = copy.deepcopy(img)
    
    if len(corners.shape) == 2:
        num_corner = len(corners)
        for i in range(num_corner):
            img = cv2.circle(img, (int(corners[i, 0]), int(corners[i, 1])), 1, (255, 0, 0), thickness=16)
            img = cv2.putText(img, '%d' % index[i], (int(corners[i, 0]), int(corners[i, 1])), cv2.FONT_HERSHEY_DUPLEX, sizet, (127,255,127), sizet, cv2.LINE_AA)
    else:
        num_corner, num_pts, _ = corners.shape
        for i in range(num_corner):
            for j in range(num_pts):
                img = cv2.circle(img, (int(corners[i, j, 0]), int(corners[i, j, 1])), 1, (255, i*5, 0), thickness=16)
                img = cv2.putText(img, '%d' % index[i * num_pts + j], (int(corners[i, j, 0]), int(corners[i, j, 1])), cv2.FONT_HERSHEY_DUPLEX, sizet, (127,255,127), sizet, cv2.LINE_AA)
    cv2.imwrite(name, img)
    
    
def visualize_corner_v2(img, corners, name='tmp.png'):
    
    img = copy.deepcopy(img)
    
    if len(corners.shape) == 2:
        num_corner = len(corners)
        for i in range(num_corner):
            img = cv2.circle(img, (int(corners[i, 0]), int(corners[i, 1])), 1, (255, 0, 0), thickness=16)
    else:
        num_corner, num_pts, _ = corners.shape
        for i in range(num_corner):
            for j in range(num_pts):
                img = cv2.circle(img, (int(corners[i, j, 0]), int(corners[i, j, 1])), 1, (255, i*5, 0), thickness=16)
    cv2.imwrite(name, img)
    
    
@torch.no_grad()
def tb_image(tb,step,group,name,images,num_vis=[4, 8],from_range=(0,1), cmap="gray"):
    images = preprocess_vis_image(images,from_range=from_range, cmap=cmap)
    num_H,num_W = num_vis
    images = images[:num_H*num_W]
    image_grid = torchvision.utils.make_grid(images[:,:3],nrow=num_W,pad_value=1.)
    if images.shape[1]==4:
        mask_grid = torchvision.utils.make_grid(images[:,3:],nrow=num_W,pad_value=1.)[:1]
        image_grid = torch.cat([image_grid,mask_grid],dim=0)
    tag = "{0}/{1}".format(group,name)
    tb.add_image(tag,image_grid,step)

def preprocess_vis_image(images,from_range=(0,1),cmap="gray"):
    min,max = from_range
    images = (images-min)/(max-min)
    images = images.clamp(min=0,max=1).cpu()
    if images.shape[1]==1:
        images = get_heatmap(images[:,0].cpu(),cmap=cmap)
    return images

def get_heatmap(gray,cmap): # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[...,:3]).permute(0,3,1,2).float() # [N,3,H,W]
    return color


@torch.no_grad()
def visualize_PSFVolume(psf_volume, mindepth, maxdepth, storepath, ep):
    
    psfV = np.float32(psf_volume.squeeze().permute(2, 1, 0).cpu().numpy())
    mind = mindepth.cpu().numpy()
    maxd = maxdepth.cpu().numpy()
    w, h, level = psfV.shape
    name = 'psfvolume_ep%04d.png' % ep
    
    X, Y, Z = np.meshgrid(np.arange(w), np.arange(h), np.linspace(mind, maxd, level))
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=psfV.flatten(),
        isomin=0.0,
        isomax=psfV.max(),
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=level // 2  # needs to be a large number for good volume rendering
    ))
    fig.show()
    fig.write_image(str(storepath / name))
    
    return fig

    
@torch.no_grad()
def visualize_samples(batch, results, stage, storepath, it, ep, rmse):
    '''
        visualize error map
    '''
    assert(len(batch['clean'][stage]) == 1)
    storepath = create_dir(storepath / 'samples')
    name = 'sample_it%04d_ep%04d_rmse_%.3f.png' % (it, ep, rmse)
    
    # read data
    clean = np.float32(batch['clean'][stage].squeeze().cpu().numpy())
    gradient = np.float32(results['gradient'].squeeze().cpu().numpy())
    blur = np.float32(results['target'].squeeze().cpu().numpy())
    convolved = np.float32(results['pred'].squeeze().cpu().numpy())
    depth = np.float32(batch['depth'][stage].squeeze().cpu().numpy())
    mask = np.float32(results['mask'].squeeze().cpu().numpy())
    
    # figure
    fig, ax = plt.subplots(1, 6, sharex='col', sharey='row', figsize=(48, 8))
    
    # save fig
    ax1 = ax[0]
    ax1.imshow(clean, aspect='equal', cmap='gray')
    plt.setp(ax1.get_xticklabels(), fontsize=8)
    plt.setp(ax1.get_yticklabels(), fontsize=8)
    ax1.set_title('latent image', fontsize=10)
    
    ax2 = ax[1]
    im2 = ax2.imshow(gradient, aspect='equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    plt.setp(ax2.get_xticklabels(), fontsize=8)
    plt.setp(ax2.get_yticklabels(), fontsize=8)
    ax2.set_title('gradient', fontsize=10)
    
    ax3 = ax[2]
    ax3.imshow(blur, aspect='equal', cmap='gray')
    plt.setp(ax3.get_xticklabels(), fontsize=8)
    plt.setp(ax3.get_yticklabels(), fontsize=8)
    ax3.set_title('blur image', fontsize=10)
    
    ax4 = ax[3]
    ax4.imshow(convolved, aspect='equal', cmap='gray')
    plt.setp(ax4.get_xticklabels(), fontsize=8)
    plt.setp(ax4.get_yticklabels(), fontsize=8)
    ax4.set_title('convolved image', fontsize=10)
    
    ax5 = ax[4]
    im5 = ax5.imshow((blur - convolved) * mask, aspect='equal')
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im5, cax=cax, orientation='vertical')
    plt.setp(ax5.get_xticklabels(), fontsize=8)
    plt.setp(ax5.get_yticklabels(), fontsize=8)
    ax5.set_title('Error map', fontsize=10)
    
    ax6 = ax[5]
    im6 = ax6.imshow(depth, aspect='auto', vmin=depth[mask > 0].min(), vmax=depth[mask > 0].max())
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im6, cax=cax, orientation='vertical')
    plt.setp(ax6.get_xticklabels(), fontsize=8)
    plt.setp(ax6.get_yticklabels(), fontsize=8)
    ax6.set_title('plane depth [mm]', fontsize=10)
        
    fig.savefig(str(storepath / name))
    plt.close()
        
        