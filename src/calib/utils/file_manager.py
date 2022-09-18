
import os
import pdb
import cv2
import shutil
from pathlib import Path
from PIL import Image
import numpy as np


# condition checker
def check_condition(condition, warner):
    if not condition:
        raise NotImplementedError(warner)
    else:
        return True


def error_handler(condition, expression, name, stop=False):
    
    '''
    :param condition: condition to check
    :param expression: error message
    :param name: location of error, use __name__ in this place
    :param stop: if condition is wrong, stop the process
    :return:
    '''

    try:
        assert condition
    except:
        if stop:
            raise NotImplementedError('%s : %s\n' % (name, expression))
        else:
            print('%s : %s\n' % (name, expression))
    
    
def option_check(value, options=None):
    error_handler(value in options, "option_check failed : %s" % value, __name__, True)


# create directory
def makedir_custom(path, opt=False):
    '''
    :param path: src path (string)
    :param opt: if exists, overwrite or not
    :return:
    '''

    if opt and path.is_dir():
        try:
            path.rmdir()
        except:
            shutil.rmtree(str(path))

    if not path.is_dir():
        path.mkdir()

    return path


def read_img(path, scale=1.0):
        
    # check ext
    img = None
    for ext in ['.JPG', '.TIF']:
        if os.path.isfile(path + ext):
            img = cv2.imread(path + ext)
            
    if scale < 1.0:
        img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    if img.shape[0] > img.shape[1]:
        return np.rot90(img, 3)
    else:
        return img
    
    
def read_img_raw(path, desired_shape=(2016, 1512)):
    
    # check ext
    for ext in ['.png', '.pgm']:
        if os.path.isfile(path + ext):
            with Image.open(path + ext) as f:
                if f.size != desired_shape:
                    f = f.resize(desired_shape)
                image = np.array(f) - 1024
                image[image < 0] = 0
                image = np.stack([np.float32(image)] * 1, axis=2) / (2 ** 14 - 1)
    
    if image.shape[0] > image.shape[1]:
        image = np.rot90(image, 3)
    
    return np.squeeze(image)


def read_dev_data(input, shared, var):
        
    device = shared['device']
    fnumber_dslr = shared['fnumber_dslr']
    focus_phone = shared['focus_phone']
    path = Path(input)
    
    info_ = np.load(str(path / 'info.npy'), allow_pickle=True).item()
        
    if device == 'dslr':  # read dslr images
        index_ = info_['fnumber_dslr'].index(fnumber_dslr)
        focal_mm = info_['focal_dslr'][index_]
        
        img_name = info_['filename_dslr'][index_]
        img_l = read_img(str(path / 'LEFT' / img_name), scale=1.0)
        img_r = read_img(str(path / 'RIGHT' / img_name), scale=1.0)
        img_c = read_img(str(path / 'CENTER' / img_name), scale=0.5)
        
    else:  # read phone images
        index_ = info_['focus_phone'].index(focus_phone)
        focal_mm = 27.0
        
        img_name = info_['filename_phone'][index_]
        img_l = read_img_raw(str(path / 'LEFT' / img_name))
        img_r = read_img_raw(str(path / 'RIGHT' / img_name))
        img_c = read_img_raw(str(path / 'CENTER' / img_name))
    
    var.put({'DPl_img': img_l, 'DPr_img': img_r, 'DPc_img': img_c, 'focal_mm': focal_mm})