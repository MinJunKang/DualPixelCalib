
import cv2
import copy, time
import torch
import numpy as np
from PIL import Image
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


def save_as_gif(images, out_path, duration=100):
    img, *imgs = [Image.fromarray(img).convert('RGBA') for img in images]
    img.save(fp=str(out_path), format='GIF', append_images=imgs, save_all=True, duration=duration, loop=0)


def visualize_PSFVolume(psf_volume, mindepth, maxdepth, storepath, tag, opacity=0.1):
    
    psfV = np.float32(psf_volume)
    w, h, level = psfV.shape
    name = 'psfvolume_%s.png' % tag
    
    X, Y, Z = np.meshgrid(np.arange(w), np.arange(h), np.linspace(mindepth, maxdepth, level))
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=psfV.flatten(),
        isomin=0.0,
        isomax=psfV.max(),
        opacity=opacity, # needs to be small to see through all surfaces
        surface_count=level  # needs to be a large number for good volume rendering
    ))
    fig.write_image(str(storepath / name))
    
    return fig
        
        