import cv2
import math
import numpy as np



def SaddleInitialize(window_size):
        
    # initialize parameters
    window_size_full = window_size * 2 + 1
    saddlekernel = np.zeros((window_size_full, window_size_full), dtype=np.float32)
    maxVal = window_size + 1
    sum = 0
    cnt = 0
    for j in range(-window_size, window_size + 1):
        for i in range(-window_size, window_size + 1):
            iidx = i + window_size
            jidx = j + window_size
            saddlekernel[jidx, iidx] = maxVal - math.sqrt(i * i + j * j)
            if saddlekernel[jidx, iidx] > 0:
                cnt += 1
            else:
                saddlekernel[jidx, iidx] = 0
            sum += saddlekernel[jidx, iidx]
    # scale kernel
    saddlekernel /= sum
    
    Amat = np.zeros((cnt, 6), dtype=np.float32)
    valid_mask = saddlekernel > 0
    
    cnt = 0
    for j in range(-window_size, window_size + 1):
        for i in range(-window_size, window_size + 1):
            iidx = i + window_size
            jidx = j + window_size
            if valid_mask[jidx, iidx]:
                Amat[cnt, 0] = i * i
                Amat[cnt, 1] = j * j
                Amat[cnt, 2] = i * j
                Amat[cnt, 3] = j
                Amat[cnt, 4] = i
                Amat[cnt, 5] = 1.0
                cnt += 1
                
    At = cv2.transpose(Amat)
    invAtAAt = cv2.invert(np.matmul(At, Amat), cv2.DECOMP_SVD)[1]  # type: ignore
    invAtAAt = np.matmul(invAtAAt, At)
    
    return saddlekernel, invAtAAt, valid_mask



def SubPixRefinement(img, corners, window_size=17, max_iter=100, step_threshold=0.001):
    '''
        Surface fitting to get subpixel accurate corners
        img : [H, W, C]
        corners : [N, 1, 2]
    '''
    
    corners = np.squeeze(corners)
    num_corner = len(corners)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = img.shape
    
    # initialize saddles
    saddlekernel, AMat, valid_mask = SaddleInitialize(window_size)
    
    # smooth circular filter
    img = cv2.filter2D(img, cv2.CV_64FC1, saddlekernel)
    
    # define saddle points
    saddle_points = np.zeros((num_corner, 6), dtype=np.float32)  # x, y, a1, a2, s, det
    saddle_points[:, :2] = corners.copy()
    bMat = np.zeros((AMat.shape[1], 1), dtype=np.float32)
    
    for n in range(num_corner):
        for iter in range(max_iter):
            px = saddle_points[n, 0]
            py = saddle_points[n, 1]
            if((px > window_size + 1) and (px < width - (window_size + 2)) and (py > window_size + 1) and (py < height - (window_size + 2))):
                x0 = int(math.floor(px))
                y0 = int(math.floor(py))
                xw = px - x0
                yw = py - y0
                
                # precompute bilinear interpolation weights
                w00 = (1.0 - xw) * (1.0 - yw)
                w01 = xw * (1.0 - yw)
                w10 = (1.0 - xw) * yw
                w11 = xw * yw
                
                # fit to local neighborhood = b vector
                cnt = 0
                for j in range(-window_size, window_size + 1):
                    y00 = y0 + j
                    y10 = y00 + 1
                    for i in range(-window_size, window_size + 1):
                        iidx = i + window_size
                        jidx = j + window_size
                        if valid_mask[jidx, iidx]:
                            x00 = x0 + i
                            x01 = x00 + 1
                            bMat[cnt, :] = img[y00, x00] * w00 + img[y00, x01] * w01 + img[y10, x00] * w10 + img[y10, x01] * w11
                            cnt += 1
                            
                # fit quadric to surface by solving LSQ
                p = np.matmul(AMat, bMat)  # [6, 1]
                
                # k5, k4, k3, k2, k1, k0
                saddle_points[n, 5] = 4.0 * p[0] * p[1] - p[2] * p[2]  # update det
                dx = (-2 * p[1] * p[4] + p[2] * p[3]) / saddle_points[n, 5]
                dy = (p[2] * p[4] - 2 * p[0] * p[3]) / saddle_points[n, 5]
                
                saddle_points[n, 0] += dx  # update x
                saddle_points[n, 1] += dy  # update y
                dx = abs(dx)
                dy = abs(dy)
                
                if (iter == max_iter) or ((step_threshold > dx) and (step_threshold > dy)):
                    # converged
                    k4mk5 = p[1] - p[0]
                    saddle_points[n, 4] = math.sqrt(p[2] * p[2] + k4mk5 * k4mk5)  # s
                    saddle_points[n, 2] = math.atan2(-p[2], k4mk5) / 2.0  # a1
                    saddle_points[n, 3] = math.acos((p[1] + p[0]) / saddle_points[n, 4]) / 2.0  # a2
                    break
                else:
                    if (saddle_points[n, 5] > 0) or (abs(saddle_points[n, 0] - corners[n, 0]) > window_size) or (abs(saddle_points[n, 1] - corners[n, 1]) > window_size):
                        saddle_points[n, 0] = saddle_points[n, 1] = np.inf
                        # diverged
                        break
            else:
                # too close to border
                saddle_points[n, 0] = saddle_points[n, 1] = np.inf
                break
    
    return saddle_points