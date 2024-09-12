
import re
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def read_img(path, scale=1.0):
    
    if path.suffix in ['pgm']:
        with Image.open(str(path)) as f:
            image = np.array(f) - 1024
            image[image < 0] = 0
            image = np.stack([np.float32(image)] * 1, axis=2) / (2 ** 14 - 1)
    else:
        img = cv2.imread(str(path))
        
    if scale < 1.0:
        img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
    return img


def read_img_focals(num_images):
    
    if num_images == 12:
        return [1.2, 1.6, 2.0, 2.5, 2.8, 3.2, 3.5, 4.0, 5.0, 5.6, 6.3, 16.0]
    elif num_images == 3:
        return [2.0, 2.8, 16.0]
    else:
        raise NotImplementedError('Number of images not supported')
    
    
def img2gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img


def read_observations(dirpath, data_type='calib', scale=0.5, target_focals=[]):
    
    assert data_type in ['calib', 'psf', 'sample']
    ext_types = ['.png', '.jpg', '.jpeg', '.JPEG', '.TIF', '.pgm']
    
    observations = dict()
    observations['device'] = 'camera'
    observations['format'] = data_type
    dirpath = Path(dirpath)
    if data_type == 'calib':
        files = sorted([path for path in dirpath.glob('*') if path.suffixes[0] in ext_types])
        assert len(files) > 0, 'Wrong format of data'
        observations['images'] = [read_img(path, scale) for path in files]
        observations['image_size'] = observations['images'][0].shape[:2]
    else:
        observations['samples'] = []
        subdirs = sorted([path for path in dirpath.glob('*') if path.is_dir()], key=lambda x: int(re.findall(r'\d+', x.stem)[0]))
        assert len(subdirs) > 0, 'Wrong format of data'
        assert len(target_focals) > 0, 'Target focals not provided'
        for subdir in tqdm(subdirs):
            files_left = sorted([path for path in (subdir / 'LEFT').glob('*') if path.suffixes[0] in ext_types])
            files_right = sorted([path for path in (subdir / 'RIGHT').glob('*') if path.suffixes[0] in ext_types])
            assert len(files_left) == len(files_right), 'Number of images in LEFT and RIGHT folders do not match'
            focals = read_img_focals(len(files_left))
            target_indices = [focals.index(focal) for focal in target_focals]
            left_images = [read_img(files_left[target_index], scale) for target_index in target_indices]
            right_images = [read_img(files_right[target_index], scale) for target_index in target_indices]
            observations['samples'].append({'images_l': left_images, 'images_r': right_images, 'focals': target_focals, 'image_size': left_images[0].shape[:2]})
    
    return observations


def read_lidar_observations(dirpath, data_type='calib'):
    
    assert data_type in ['calib', 'sample']
    
    observations = dict()
    observations['device'] = 'lidar'
    observations['format'] = data_type
    dirpath = Path(dirpath)
    if data_type == 'calib':
        subdirs = sorted([path for path in dirpath.glob('*') if path.is_dir()], key=lambda x: int(re.findall(r'\d+', x.stem)[0]))
        assert len(subdirs) > 0, 'Wrong format of data'
        observations['images'] = [read_img(subdir / 'LIDAR_IMG.png') for subdir in subdirs]
        observations['image_size'] = observations['images'][0].shape[:2]
    else:
        observations['samples'] = []
        subdirs = sorted([path for path in dirpath.glob('*') if path.is_dir()], key=lambda x: int(re.findall(r'\d+', x.stem)[0]))
        assert len(subdirs) > 0, 'Wrong format of data'
        for subdir in tqdm(subdirs):
            images = read_img(subdir / 'LIDAR_IMG.png')
            depths = np.load(subdir / 'LIDAR_DEPTH.npy')
            masks = np.load(subdir / 'LIDAR_MASK.npy')
            observations['samples'].append({'images': images, 'depths': depths, 'masks': masks, 'image_size': images.shape[:2], 'depth_size': depths.shape[:2]})

    return observations