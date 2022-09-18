
from yacs.config import CfgNode as CN

cfg = CN()
cfg.blobParams = CN()

cfg.blobParams.minthres = 0.10
cfg.blobParams.maxthres = 0.95
cfg.blobParams.filterByArea = True
cfg.blobParams.maxArea = 3500

cfg.blobParams.filterByCircularity = True
cfg.blobParams.minCircularity = 0.1
cfg.blobParams.filterByConvexity = True
cfg.blobParams.minConvexity = 0.87

cfg.blobParams.filterByInertia = True
cfg.blobParams.minInertiaRatio = 0.01