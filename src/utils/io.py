
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def read_img(path, scale=1.0):
    
    if path.suffix in ['pgm']:
        with Image.open(str(path)) as f:
            if f.size != desired_shape:
                f = f.resize(desired_shape)
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
    else:
        raise NotImplementedError('Number of images not supported')
    
    
def img2gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img


def read_observations(dirpath, data_type='calib', scale=0.5):
    
    assert data_type in ['calib', 'psf']
    ext_types = ['.png', '.jpg', '.jpeg', '.JPEG', '.TIF', '.pgm']
    
    observations = dict()
    if data_type == 'calib':
        observations['format'] = 'calib'
        dirpath = Path(dirpath)
        files = sorted([path for path in dirpath.glob('*') if path.suffixes[0] in ext_types])
        assert len(files) > 0, 'Wrong format of data'
        observations['images'] = [read_img(path, scale) for path in files]
        observations['image_size'] = observations['images'][0].shape[:2]
    else:
        observations['format'] = 'psf'
        dirpath = Path(dirpath)
        observations['samples'] = []
        subdirs = sorted([path for path in dirpath.glob('*') if path.is_dir()])
        assert len(subdirs) > 0, 'Wrong format of data'
        for subdir in tqdm(subdirs):
            files_left = sorted([path for path in (subdir / 'LEFT').glob('*') if path.suffixes[0] in ext_types])
            files_right = sorted([path for path in (subdir / 'RIGHT').glob('*') if path.suffixes[0] in ext_types])
            left_images = [read_img(path, scale) for path in files_left]
            right_images = [read_img(path, scale) for path in files_right]
            assert len(left_images) == len(right_images), 'Number of images in LEFT and RIGHT folders do not match'
            focals = read_img_focals(len(left_images))
            observations['samples'].append({'images_l': left_images, 'images_r': right_images, 'focals': focals, 'image_size': left_images[0].shape[:2]})
    return observations