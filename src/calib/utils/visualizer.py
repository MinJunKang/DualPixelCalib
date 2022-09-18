
import cv2


def visualize_corner(img, corners, index, name='tmp.png', sizet=4):
    
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