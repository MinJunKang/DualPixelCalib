
import cv2
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .file_manager import makedir_custom


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
                img = cv2.putText(img, '%d' % index[i], (int(corners[i, j, 0]), int(corners[i, j, 1])), cv2.FONT_HERSHEY_DUPLEX, sizet, (127,255,127), sizet, cv2.LINE_AA)
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
    
    
def visualize_samples(batch, results, storepath, index, samplename):
    '''
        visualize error map
    '''
    
    # will only take first batch
    with torch.no_grad():
        clean = normalize(batch['clean'], batch['depth'] > 0).squeeze().cpu().numpy()
        blur = normalize(batch['blur'], batch['depth'] > 0).squeeze().cpu().numpy()
        gradient = np.float32(results['gradient'].squeeze().cpu().numpy())
        depth = np.float32(batch['depth'].squeeze().cpu().numpy())
        convolved = np.float32(results['convolved'].squeeze().cpu().numpy())
        mask = depth > 0
        
        storepath = makedir_custom(storepath / ('epoch%04d_samples' % index), False)
        
        # save fig
        name = 'psf_epoch%04d_%s.png' % (index, samplename)
        fig, ax = plt.subplots(1, 6, sharex='col', sharey='row', figsize=(48, 8))
        fig.suptitle('%s of Epoch %02d' % (samplename, index))
        
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
        im5 = ax5.imshow((blur - convolved), aspect='equal')
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im5, cax=cax, orientation='vertical')
        plt.setp(ax5.get_xticklabels(), fontsize=8)
        plt.setp(ax5.get_yticklabels(), fontsize=8)
        ax5.set_title('Error map', fontsize=10)
        
        ax6 = ax[5]
        im6 = ax6.imshow(depth, aspect='auto', vmin=depth[mask].min(), vmax=depth[mask].max())
        divider = make_axes_locatable(ax6)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im6, cax=cax, orientation='vertical')
        plt.setp(ax6.get_xticklabels(), fontsize=8)
        plt.setp(ax6.get_yticklabels(), fontsize=8)
        ax6.set_title('plane depth [mm]', fontsize=10)
            
        fig.savefig(str(storepath / name))
        plt.close()