
import cv2
import math
import argparse
import numpy as np
from pathlib import Path
from src.calib.utils.aruco import ARUCO_DICT
from src.calib.utils.geometry import convert2homography
from src.calib.utils.file_manager import makedir_custom, check_condition



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


def initialize(board_info):
    
    num_x_square = board_info.num_x_square
    num_y_square = board_info.num_y_square
    length_square = board_info.length_square
    length_marker = board_info.length_marker
    inner_offset = board_info.inner_offset
    res_x = int(board_info.res_x)
    res_y = int(board_info.res_y)
    offset = board_info.offset

    dictpattern = cv2.aruco.Dictionary_get(ARUCO_DICT[board_info.aruco_type])
    charuco = cv2.aruco.CharucoBoard_create(num_x_square, num_y_square, length_square, length_marker, dictpattern)
    '''
    squarex : number of chessboard squares in X direction
    squarey : number of chessboard squares in Y direction

    squareLength : chessboard square side length (in meters)
    markerLength : marker side length (in meters)
    '''
    iboard = charuco.draw((res_x, res_y), 0, 0)
    
    # basic information
    block_size_px = min(res_x // num_x_square, res_y // num_y_square)
    pad_x, pad_y = (res_x - block_size_px * num_x_square) // 2, (res_y - block_size_px * num_y_square) // 2
    
    # make inner area as black
    iboard[pad_y + block_size_px * inner_offset - 1:res_y - block_size_px * inner_offset - pad_y, pad_x + block_size_px * inner_offset - 1:res_x - block_size_px * inner_offset - pad_x] = 0
    
    # padding edges
    if offset >= 1:
        iboard = np.pad(iboard, ((offset, offset), (offset, offset)), 'constant', constant_values=255)
        offset_x, offset_y = pad_x + offset, pad_y + offset
    else:
        offset_x, offset_y = pad_x, pad_y
    offsets = [offset_x, offset_y, block_size_px]

    return iboard, offsets


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


def create_full_board(board_info, size_square_mm, scale=1.0):

    # create charuco board
    board_info.res_x = int(board_info.res_x * scale)
    board_info.res_y = int(board_info.res_y * scale)
    iboard, offsets = initialize(board_info)

    # draw white circles in charuco board
    fboard, rec_property, cir_property = draw_circles(iboard, offsets, board_info)

    # information of board
    offset_x, offset_y, block_size_px = offsets
    
    # cropped template image
    cropped_template = fboard[offset_y:-offset_y, offset_x:-offset_x]
    
    # prepare 2D/3D corner points
    x_pts = np.linspace(0, board_info.num_x_square, board_info.num_x_square + 1)
    y_pts = np.linspace(board_info.num_y_square, 0, board_info.num_y_square + 1)
    _2d_corner = np.stack(np.meshgrid(x_pts, y_pts), axis=-1)
    _2d_world = _2d_corner * size_square_mm
    _2d_template = _2d_corner.copy()
    _2d_template[:, :, 0] *= block_size_px
    _2d_template[:, :, 1] *= block_size_px
    _3d_world = convert2homography(_2d_world, 0.0)
    _3d_template = convert2homography(_2d_template, 0.0)
    
    # prepare 2D/3D patches' corner points
    _2d_patch_corner = cir_property[0].copy()
    _2d_patch_corner[:, :, :, :, 0] = (_2d_patch_corner[:, :, :, :, 0] - offset_x + 0.5) / block_size_px
    _2d_patch_corner[:, :, :, :, 1] = (_2d_patch_corner[:, :, :, :, 1] - offset_y + 0.5) / block_size_px
    _2d_patch_world = _2d_patch_corner * size_square_mm
    _2d_patch_template = _2d_patch_corner.copy()
    _2d_patch_template[:, :, :, :, 0] *= block_size_px
    _2d_patch_template[:, :, :, :, 1] *= block_size_px
    _3d_patch_world = convert2homography(_2d_patch_world, 0.0)
    _3d_patch_template = convert2homography(_2d_patch_template, 0.0)
    
    # prepare circle's coordinate
    _2d_circle = cir_property[1].copy()
    _2d_circle_radius = cir_property[2].copy()
    _2d_circle[:, :, :, 0] = (_2d_circle[:, :, :, 0] - offset_x + 0.5) / block_size_px
    _2d_circle[:, :, :, 1] = (_2d_circle[:, :, :, 1] - offset_y + 0.5) / block_size_px
    _2d_circle_world = _2d_circle * size_square_mm
    _2d_circle_template = _2d_circle.copy()
    _2d_circle_template[:, :, :, 0] *= block_size_px
    _2d_circle_template[:, :, :, 1] *= block_size_px
    _3d_circle_world = convert2homography(_2d_circle_world, 0.0)
    _3d_circle_template = convert2homography(_2d_circle_template, 0.0)
    
    # put into dict
    board_info_ = dict()
    board_info_['num_grid'] = board_info.num_grid
    board_info_['template'] = cropped_template
    board_info_['size_px'] = block_size_px
    board_info_['size_py'] = block_size_px
    board_info_['num_x'] = board_info.num_x_square
    board_info_['num_y'] = board_info.num_y_square
    board_info_['inner_offset'] = board_info.inner_offset
    board_info_['rec2d_world'] = _2d_world
    board_info_['rec2d_template'] = _2d_template
    board_info_['rec3d_world'] = _3d_world
    board_info_['rec3d_template'] = _3d_template
    
    board_info_['cir2d_world'] = _2d_circle_world
    board_info_['cir2d_template'] = _2d_circle_template
    board_info_['cir3d_world'] = _3d_circle_world
    board_info_['cir3d_template'] = _3d_circle_template
    board_info_['radius'] = _2d_circle_radius
    
    board_info_['patch2d_world'] = _2d_patch_world
    board_info_['patch2d_template'] = _2d_patch_template
    board_info_['patch3d_world'] = _3d_patch_world
    board_info_['patch3d_template'] = _3d_patch_template
    
    # board dictionary info
    dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[board_info.aruco_type])
    arucoParams = cv2.aruco.DetectorParameters_create()
    arucoParams.adaptiveThreshConstant = 1.0
    arucoParams.detectInvertedMarker = True
    board = cv2.aruco.CharucoBoard_create(board_info.num_x_square, board_info.num_y_square, 
                                          board_info.length_square, board_info.length_marker, 
                                          dictionary)
    board_info_['dictionary'] = dictionary
    board_info_['board'] = board
    board_info_['scale'] = scale

    return board_info_


def main():
    
    # parsing arguments
    parser = argparse.ArgumentParser(description='Configuration : board creater')
    parser.add_argument('--num_x_square', type=int, default=7, help='number of chessboard squares in X direction')
    parser.add_argument('--num_y_square', type=int, default=5, help='number of chessboard squares in Y direction')
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
    iboard, offsets = initialize(info)

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

    path = makedir_custom(Path('./boards'), False)
    path = makedir_custom(path / info.boardname, True)

    # draw board
    cv2.imwrite(str(path / 'board.png'), fboard)
    np.save(str(path / 'board_info.npy'), info)

if __name__ == '__main__':
    main()
    
    
