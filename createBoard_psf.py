
import cv2
import math
import argparse
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict
from src.utils.aruco import ARUCO_DICT
from src.utils.math import convert2homography
from src.utils.base import check_condition, create_dir


def inner_rec_corners(num_x_square, num_y_square, inner_offset=1):
    
    # corners index group
    corners1_x_idx = np.array(range(inner_offset, num_x_square - inner_offset, 1))
    corners2_x_idx = np.array(range(inner_offset + 1, num_x_square + (1 - inner_offset), 1))
    corners1_y_idx = np.flip(np.array(range(inner_offset, num_y_square - inner_offset, 1)))
    corners2_y_idx = np.flip(np.array(range(inner_offset + 1, num_y_square + (1 - inner_offset), 1)))
    
    # corners
    '''
        corner3      corner4


        corner1      corner2
    '''

    # corners
    corner1 = np.stack(np.meshgrid(corners1_x_idx, corners2_y_idx), axis=-1)
    corner2 = np.stack(np.meshgrid(corners2_x_idx, corners2_y_idx), axis=-1)
    corner3 = np.stack(np.meshgrid(corners1_x_idx, corners1_y_idx), axis=-1)
    corner4 = np.stack(np.meshgrid(corners2_x_idx, corners1_y_idx), axis=-1)

    corner1 = np.reshape(corner1, [-1, 1, 2])
    corner2 = np.reshape(corner2, [-1, 1, 2])
    corner3 = np.reshape(corner3, [-1, 1, 2])
    corner4 = np.reshape(corner4, [-1, 1, 2])
    
    corners_loc = np.concatenate([corner1, corner2, corner3, corner4], axis=1)  # [N, 4, 2]
    
    # get corners' aruco idx
    corners_idx = corners_loc[:, :, 0] + (num_y_square - corners_loc[:, :, 1] - 1) * (num_x_square)

    return corners_loc, corners_idx


def get_charuco_obj_corners(charuco, num_y_square, num_x_square, length_square):
    pattern_ids = charuco.getIds()
    pattern_corners = np.stack(charuco.getObjPoints(), axis=0) / length_square
    idx_start = 0
    syn_board_ids = np.ones((num_y_square, num_x_square), dtype=np.int32) * (-1)
    syn_board_corners = np.zeros((num_y_square, num_x_square, 4, 2), dtype=np.float32)
    for i in range(num_y_square):
        if i % 2 == 0:
            for j in range(num_x_square // 2):
                syn_board_ids[i, j * 2 + 1] = pattern_ids[idx_start]
                syn_board_corners[i, j * 2 + 1] = pattern_corners[idx_start][:, :2]
                idx_start += 1
        else:
            for j in range(num_x_square // 2 + 1):
                syn_board_ids[i, j * 2] = pattern_ids[idx_start]
                syn_board_corners[i, j * 2] = pattern_corners[idx_start][:, :2]
                idx_start += 1
    return syn_board_corners, syn_board_ids


def initialize(board_info):
    
    num_x_square = board_info.num_x_square
    num_y_square = board_info.num_y_square
    length_square = board_info.length_square
    length_marker = board_info.length_marker
    inner_offset = board_info.inner_offset
    res_x = int(board_info.res_x)
    res_y = int(board_info.res_y)
    offset = board_info.offset

    dictpattern = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[board_info.aruco_type])
    charuco = cv2.aruco.CharucoBoard((num_x_square, num_y_square), length_square, length_marker, dictpattern)
    charuco_obj_corners, charuco_obj_ids = get_charuco_obj_corners(charuco, num_y_square, num_x_square, length_square)
    '''
    squarex : number of chessboard squares in X direction
    squarey : number of chessboard squares in Y direction

    squareLength : chessboard square side length (in meters)
    markerLength : marker side length (in meters)
    '''
    # charuco corners
    '''
        corner1      corner2


        corner4      corner3
    '''
    iboard = charuco.generateImage((res_x, res_y), 0, 0)
    
    # basic information
    block_size_px = min(res_x // num_x_square, res_y // num_y_square)
    pad_x, pad_y = (res_x - block_size_px * num_x_square) // 2, (res_y - block_size_px * num_y_square) // 2
    
    # make inner area as black
    iboard[pad_y + block_size_px * inner_offset - 1:res_y - block_size_px * inner_offset - pad_y, pad_x + block_size_px * inner_offset - 1:res_x - block_size_px * inner_offset - pad_x] = 0
    charuco_obj_corners[inner_offset:num_y_square - inner_offset, inner_offset:num_x_square - inner_offset] = 0
    charuco_obj_ids[inner_offset:num_y_square - inner_offset, inner_offset:num_x_square - inner_offset] = -1
    
    # padding edges
    if offset >= 1:
        iboard = np.pad(iboard, ((offset, offset), (offset, offset)), 'constant', constant_values=255)
        offset_x, offset_y = pad_x + offset, pad_y + offset
    else:
        offset_x, offset_y = pad_x, pad_y
    offsets = [offset_x, offset_y, block_size_px]
    
    # get charuco obj information
    mask = charuco_obj_ids >= 0
    charuco_obj_infos = {'corners': charuco_obj_corners[mask], 'ids': charuco_obj_ids[mask]}

    return iboard, offsets, charuco_obj_infos


def find_circle_centers(rec_corners, num_grid, radius):
    
    # define grid by using num_grid
    grid_x, grid_y = np.meshgrid(np.arange(num_grid), np.arange(num_grid))
    
    # patch number
    num_patch = len(rec_corners)
    
    # circles center and radius
    circles_patch = np.zeros((num_patch, num_grid, num_grid, 4, 2))
    circles_center = np.zeros((num_patch, num_grid, num_grid, 2))
    circles_radius = np.zeros((num_patch, num_grid, num_grid, 1))
    
    for i in range(num_patch):
        
        # find distance between corners
        start = rec_corners[i, 2, :]
        dist_x = rec_corners[i, 1, 0] - rec_corners[i, 0, 0]
        dist_y = rec_corners[i, 0, 1] - rec_corners[i, 2, 1]
        patchsize_x = dist_x / num_grid
        patchsize_y = dist_y / num_grid
        minpatchsize = min(patchsize_x, patchsize_y)
        
        # grid to scale
        corner3_x = grid_x * patchsize_x + start[0]
        corner3_y = grid_y * patchsize_y + start[1]
        corner2_x = corner3_x + patchsize_x
        corner2_y = corner3_y + patchsize_y
        corner1_x = corner3_x
        corner1_y = corner3_y + patchsize_y
        corner4_x = corner3_x + patchsize_x
        corner4_y = corner3_y
        corner1 = np.stack([corner1_x, corner1_y], axis=-1)
        corner2 = np.stack([corner2_x, corner2_y], axis=-1)
        corner3 = np.stack([corner3_x, corner3_y], axis=-1)
        corner4 = np.stack([corner4_x, corner4_y], axis=-1)
        
        # corner information
        circles_patch[i, :, :, 0, :] = corner1
        circles_patch[i, :, :, 1, :] = corner2
        circles_patch[i, :, :, 2, :] = corner3
        circles_patch[i, :, :, 3, :] = corner4
        
        # circles information
        circles_center[i] = (corner1 + corner2 + corner3 + corner4) / 4.0
        circles_radius[i] = minpatchsize * radius
        
    return circles_patch, circles_center, circles_radius


def draw_circles(iboard, offsets, board_info):
    
    shift = board_info.shift
    radius = board_info.radius
    num_grid = board_info.num_grid
    num_x_square = board_info.num_x_square
    num_y_square = board_info.num_y_square
    length_square = board_info.length_square
    inner_offset = board_info.inner_offset
    
    # corners of inner rectangular
    rec_corners, rec_corners_idx = inner_rec_corners(num_x_square, num_y_square, inner_offset)
    Nrectangular = len(rec_corners)

    # center of circles in black rectangular
    rec_corners_scale = rec_corners.copy()  # [N, 4, 2]
    rec_corners_scale[:, :, 0] = rec_corners_scale[:, :, 0] * offsets[-1] + offsets[0] - 0.5
    rec_corners_scale[:, :, 1] = rec_corners_scale[:, :, 1] * offsets[-1] + offsets[1] - 0.5
    circles_patch, circles_center, circles_radius = find_circle_centers(rec_corners_scale, num_grid, radius / length_square)  # [y, x], [N, 2]
    
    # draw circles in iboard
    factor = (1 << shift)
    for i in range(Nrectangular):
        for n in range(num_grid):
            for m in range(num_grid):
                # subpixel drawing circle
                center_x_subpix = int(circles_center[i, n, m, 0] * factor + 0.5)  # decimal scaling for subpixel precision
                center_y_subpix = int(circles_center[i, n, m, 1] * factor + 0.5)  # decimal scaling for subpixel precision
                radius_subpix = int(circles_radius[i, n, m, 0] * factor + 0.5)
                iboard = cv2.circle(img=iboard, center=(center_x_subpix, center_y_subpix), radius=radius_subpix, color=(255, 255, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA, shift=shift)
        
    return iboard, (rec_corners, rec_corners_scale, rec_corners_idx), (circles_patch, circles_center, circles_radius)


def get_all_paired_line_finding_circle(corners, corners_id, circles_center):
    
    # get all possible lines
    lines = []
    corners_all = corners.reshape(-1, 2)
    for i in range(len(corners_all)):
        for j in range(i + 1, len(corners_all)):
            x1, y1 = corners_all[i]
            x2, y2 = corners_all[j]
            if x1 != x2 and y1 != y2:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
            else:
                continue
            lines.append(np.array([corners_id[i // 4], i % 4, corners_id[j // 4], j % 4, m, b]))
    
    pairs_all = []  # (line1_corner_id, line1_corner_idx, line2_corner_id, line2_corner_idx, circle_idx)
    circles_all = circles_center.reshape(-1, 2)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            m1, b1 = lines[i][4], lines[i][5]
            m2, b2 = lines[j][4], lines[j][5]
            if m1 != m2:
                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1
                dists = np.linalg.norm(circles_all - np.array([x, y])[None], axis=1)
                mask = dists < 1.0
                if mask.any():
                    indices = np.where(mask)[0].tolist()
                    for idx in indices:
                        lines_property = np.concatenate([lines[i][:4], lines[j][:4], [idx]]).astype('uint32')
                        pairs_all.append(lines_property)
            else:
                continue
    return np.stack(pairs_all, axis=0)


def create_full_board(board_info, size_square_mm, scale=1.0):
    board_info = edict(board_info)

    # create charuco board
    board_info.res_x = int(board_info.res_x * scale)
    board_info.res_y = int(board_info.res_y * scale)
    iboard, offsets, charuco_obj_infos = initialize(board_info)

    # draw white circles in charuco board
    fboard, rec_property, cir_property = draw_circles(iboard, offsets, board_info)

    # information of board
    offset_x, offset_y, block_size_px = offsets
    
    # cropped template image
    cropped_template = fboard[offset_y:-offset_y, offset_x:-offset_x]
    
    # prepare 2D/3D properties
    charuco_obj_3d = charuco_obj_infos['corners'] * size_square_mm
    charuco_obj_2d = charuco_obj_infos['corners'] * block_size_px
    circles_center, circles_radius = cir_property[1].copy(), cir_property[2].copy()
    circles_center[:, :, :, 0] = (circles_center[:, :, :, 0] - offset_x + 0.5) / block_size_px
    circles_center[:, :, :, 1] = (circles_center[:, :, :, 1] - offset_y + 0.5) / block_size_px
    circles_center_2d = circles_center * block_size_px
    circles_center_3d = circles_center * size_square_mm
    
    # convert to homography coordinate
    charuco_obj_2d = convert2homography(charuco_obj_2d, 0.0)
    charuco_obj_3d = convert2homography(charuco_obj_3d, 0.0)
    circles_center_2d = convert2homography(circles_center_2d, 0.0)
    circles_center_3d = convert2homography(circles_center_3d, 0.0)
    
    # put into dict
    board_info_ = dict()
    board_info_['num_grid'] = board_info.num_grid
    board_info_['template'] = cropped_template
    board_info_['size_px'] = block_size_px
    board_info_['num_x'] = board_info.num_x_square
    board_info_['num_y'] = board_info.num_y_square
    board_info_['obj_corners_id'] = charuco_obj_infos['ids']
    board_info_['obj_corners_2d'] = charuco_obj_2d
    board_info_['obj_corners_3d'] = charuco_obj_3d
    
    board_info_['cir2d'] = circles_center_2d
    board_info_['cir3d'] = circles_center_3d
    board_info_['radius'] = circles_radius
    
    # board dictionary info
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[board_info.aruco_type])
    arucoParams = cv2.aruco.DetectorParameters()
    arucoParams.adaptiveThreshConstant = 1.0
    arucoParams.detectInvertedMarker = True
    board = cv2.aruco.CharucoBoard((board_info.num_x_square, board_info.num_y_square), 
                                   board_info.length_square, board_info.length_marker, 
                                   dictionary)
    board_info_['dictionary'] = dictionary
    board_info_['board'] = board
    board_info_['scale'] = scale

    return board_info_


def main():
    
    # parsing arguments
    parser = argparse.ArgumentParser(description='Configuration : board creater')
    parser.add_argument('--num_x_square', type=int, default=9, help='number of chessboard squares in X direction')
    parser.add_argument('--num_y_square', type=int, default=7, help='number of chessboard squares in Y direction')
    parser.add_argument('--num_grid', type=int, default=2, help='number of grid in specific direction')
    parser.add_argument('--length_square', type=float, default=2.0, help='number of chessboard squares in Y direction')
    parser.add_argument('--length_marker', type=float, default=1.6, help='number of chessboard squares in Y direction')
    parser.add_argument('--resolution', type=int, default=3072, help='base resolution')  # 768
    parser.add_argument('--radius', type=float, default=0.15, help='radius of circle')
    parser.add_argument('--offset', type=int, default=8, help='padding number')
    parser.add_argument('--inner_offset', type=int, default=1, help='padding number')
    parser.add_argument('--shift', type=int, default=16, help='for subpixel accuracy circle drawing')
    parser.add_argument('--aruco_type', type=str, default='DICT_4X4_50', help='aruco marker')
    parser.add_argument('--boardname', type=str, required=True, help='board name')

    info = parser.parse_args()
    max_num = max(info.num_x_square, info.num_y_square)
    info.res_x = int(info.resolution * info.num_x_square / max_num // 48) * 48
    info.res_y = int(info.resolution * info.num_y_square / max_num // 48) * 48

    # check condition
    check_condition((info.num_x_square > 0) & (info.num_y_square > 0), 'invalid number of square')
    check_condition(info.length_marker <= info.length_square, 'square length must be bigger than marker length')
    check_condition(info.radius <= info.length_square * 0.5, 'radius of circle should be lower than half of square length')
    check_condition(info.shift >= 0, 'shift value should be bigger than 0')

    # create charuco board
    iboard, offsets, _ = initialize(info)

    # draw white circles in charuco board
    if info.radius > 0:
        fboard, rec_property, cir_property = draw_circles(iboard, offsets, info)
    else:
        fboard, rec_property, cir_property = iboard, None, None

    # save the board information
    saved_info = vars(info)
    saved_info['rec_property'] = rec_property
    saved_info['cir_property'] = cir_property
    saved_info['offsets'] = offsets

    path = create_dir(Path('./boards'), True)
    path = create_dir(path / info.boardname, False)

    # draw board
    cv2.imwrite(str(path / 'board.png'), fboard)
    np.save(str(path / 'board_info.npy'), saved_info)

if __name__ == '__main__':
    main()
    
    
