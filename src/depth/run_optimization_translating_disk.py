import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import cv2
import tqdm
from scipy.signal import convolve2d
from scipy.optimize import minimize, least_squares, Bounds
import multiprocessing
from multiprocessing import Process, Value, Array
import ctypes as c


def check_shift_direction(img1, img2):
    img_h, img_w = img1.shape

    divide_4 = img_w % 4
    img1 = img1[:, divide_4:]
    img2 = img2[:, divide_4:]
    img_w = img_w - divide_4

    win = img_w // 2
    r = img_w // 4
    ssd_mat = np.zeros(win + 1)
    for j in range(win + 1):
        ssd_mat[j] = np.sum(np.sum((img1[:, r:-r] - img2[:, j:j + win]) ** 2))
    ind = np.argmin(ssd_mat)
    if ind > win // 2 + 1:
        myflag = 1
    else:
        myflag = 0
    return myflag    


def solver_translating_disk_kernel(kersize, patlg, patrg, border):
    # do a simple SSD search to find direction of shift
    # initializing in the right direction helps speed up the optimization
    myflag = check_shift_direction(patlg, patrg)
    # options = {'disp': True, 'ftol': 0, 'eps': 5, 'maxiter': 100}
    # options = {'disp': False, 'ftol': 1e-6, 'maxiter': 1000} ##Powell
    # options = {'disp': True, 'gtol': 1e-6, 'maxiter': 1000} ##BFGS
    options = {'disp': True, 'gtol': 1e-10, 'maxiter': 100, 'verbose':3}#, 'initial_tr_radius': 0.1}
    # ineq_cons = {'type': 'ineq',
    #              'fun' : lambda x: 1e-6 - myfun(x, patlg, patrg, kersize, border)}
    A = None
    b = None
    Aeq = None
    beq = None
    lb = -(kersize-1)/2
    ub = (kersize-1)/2
    bounds = Bounds(lb, ub)
    if myflag == 1:
        x0 = -1
    else:
        x0 = 1
    nonlcon = None
    # res = minimize(myfun, x0, args=(patlg, patrg, kersize, border), method='BFGS', bounds=bounds, options=options) #, constraints=[ineq_cons]) 
    # res = minimize(myfun, x0, args=(patlg, patrg, kersize, border), method='Powell', bounds=bounds, options=options) #, constraints=[ineq_cons]) 
    res = minimize(myfun, x0, args=(patlg, patrg, kersize, border), method='trust-constr', bounds=bounds, options=options) #, constraints=[ineq_cons]) 
    # res = minimize(myfun, x0, args=(patlg, patrg, kersize, border), method='Powell', bounds=bounds, options=options) #, constraints=[ineq_cons]) 
    # res = minimize(myfun, x0, args=(patlg, patrg, kersize, border), method='SLSQP', bounds=[(lb, ub)], options=options,constraints=nonlcon)# constraints=[ineq_cons]) 
    # res = least_squares(myfun, x0, bounds =(lb, ub), args=(patlg, patrg, kersize, border), gtol=None)
    x = res.x[0]
    fval = res.fun
    ker_est = ker_disk(x, kersize)

    return x, ker_est, fval

def myfun(xe, patlg, patrg, kersize, border):
    h = ker_disk(xe, kersize)
    # only flipped kernel symmetry cost
    l = convolve2d(patlg, np.fliplr(h), mode='same')
    r = convolve2d(patrg, h, mode='same')
    l = l[border:-border, border:-border]
    r = r[border:-border, border:-border]
    err = (l - r) / 255
    xerr = np.sum(err**2) / np.size(err)
    return xerr


def ker_disk(kersig, kersize):
    circ = np.zeros((kersize, kersize))
    circ_x, circ_y = np.meshgrid(np.arange(kersize), np.arange(kersize))
    mid = (kersize//2 + 1)
    circ_bool = ((circ_x - mid)**2 + (circ_y - mid)**2) < kersig**2
    circ[circ_bool] = 1
    circ = circ/circ.sum()
    # refcirc = np.zeros((kersize, kersize))
    # refcirc[(kersize - 2 * radius) // 2:kersize - (kersize - 2 * radius) // 2 + 1,
            # (kersize - 2 * radius) // 2:kersize - (kersize - 2 * radius) // 2 + 1] = circ
    dist_array = np.linspace(0, 2 * np.abs(int(kersig)) + 1, 2 * np.abs(int(kersig)) + 1 + 1)
    diskker = np.zeros((kersize, kersize))
    for i in dist_array:
        diskker += circ * np.roll(circ, int(np.sign(kersig) * i), axis=1)
    kerout = 0.5 * diskker / np.sum(diskker)
    return kerout

def initpool(arr1, arr2, arr3):
    global array1, array2, array3
    array1 = arr1
    array2 = arr2
    array3 = arr3

def solver_multithread(ker_size, imglg, imgrg, imgcg, border, i, j, m, img_h, img_w):

    sb = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    ker_sig, _, fval = solver_translating_disk_kernel(ker_size, imglg[i-m:i+m+1, j-m:j+m+1], imgrg[i-m:i+m+1, j-m:j+m+1], border)

    sobel_val = convolve2d(imgcg[i-m:i+m, j-m:j+m], sb, 'same')
    
    out_map = np.frombuffer(array1.get_obj(), dtype=np.float64).reshape(img_h, img_w)
    out_map[i-m:i+m, j-m:j+m] = ker_sig

    out_fval = np.frombuffer(array2.get_obj(), dtype=np.float64).reshape(img_h, img_w)
    out_fval[i-m:i+m, j-m:j+m] = fval

    out_sobel = np.frombuffer(array3.get_obj(), dtype=np.float64).reshape(img_h, img_w)
    out_sobel[i-m:i+m, j-m:j+m] = sobel_val
    print(i, j, ker_sig)

def run_optimization_translating_disk_kernel(patch_size, ker_size, imglg, imgrg, imgcg, border, stride):
    
    img_h, img_w = imglg.shape

    m = patch_size//2
    mids = stride//2
    #
    # out_map = np.zeros([img_h, img_w])
    # out_fval = np.zeros([img_h, img_w])
    # out_sobel = np.zeros([img_h, img_w])
    
    out_map = Array(c.c_double, img_h*img_w)
    out_map_np = np.frombuffer(out_map.get_obj()).reshape(img_h, img_w)

    out_fval = Array(c.c_double, img_h*img_w)
    out_fval_np = np.frombuffer(out_fval.get_obj()).reshape(img_h, img_w)
    out_sobel = Array(c.c_double, img_h*img_w)
    out_sobel_np = np.frombuffer(out_sobel.get_obj()).reshape(img_h, img_w)
    
    np.copyto(out_map_np, np.zeros([img_h, img_w]))
    np.copyto(out_fval_np, np.zeros([img_h, img_w]))
    np.copyto(out_sobel_np, np.zeros([img_h, img_w]))


    # sb = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # sb = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    pool = multiprocessing.Pool(processes=1, initializer=initpool, initargs=(out_map, out_fval, out_sobel))
    result_p_list = []
    for i in range(m, img_h-m, stride):
        for j in range(m, img_w-m, stride):
            result_p_list.append(pool.apply_async(solver_multithread, (ker_size, imglg, imgrg, imgcg, border, i, j, m, img_h, img_w)))

    result_p_list = [r.get() for r in result_p_list]
    pool.close()
    pool.join()

    map_out = np.frombuffer(out_map.get_obj()).reshape(img_h, img_w)
    fval_out = np.frombuffer(out_fval.get_obj()).reshape(img_h, img_w)
    sobel_out = np.frombuffer(out_sobel.get_obj()).reshape(img_h, img_w)

    return map_out, fval_out, sobel_out

if __name__ == "__main__":
    input_img_L = cv2.imread("./data/Punnappurath_ICCP_2020/010_L.jpg")
    input_img_L = cv2.cvtColor(input_img_L, cv2.COLOR_BGR2GRAY)
    input_img_L = cv2.resize(input_img_L, None, fx=0.5, fy=0.5)

    input_img_R = cv2.imread("./data/Punnappurath_ICCP_2020/010_R.jpg")
    input_img_R = cv2.cvtColor(input_img_R, cv2.COLOR_BGR2GRAY)
    input_img_R = cv2.resize(input_img_R, None, fx=0.5, fy=0.5)

    input_img_C = cv2.imread("./data/Punnappurath_ICCP_2020/010_B.jpg")
    input_img_C = cv2.cvtColor(input_img_C, cv2.COLOR_BGR2GRAY)
    input_img_C = cv2.resize(input_img_C, None, fx=0.5, fy=0.5)

    patch_size = 111
    kernel_size = 41
    stride = 33
    border = 25
    out_map, fval, sobel = run_optimization_translating_disk_kernel(patch_size, kernel_size, input_img_L, input_img_R, input_img_C, border, stride)