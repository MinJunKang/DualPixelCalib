
# User Guidance

## 1. Installation
```
git clone https://github.com/MinJunKang/DualPixelCalib.git -b minjun
pip install -r requirements.txt
```

## 2. Captured Dataset
### Set path of CALIB_FOLDER and PSF_FOLDER at configs/paths/default.yaml
```
- CALIB_FOLDER
    - F06_av16
        - *.TIF
    - F15_av16
        - *.TIF
- PSF_FOLDER
    - F06
    - F06_v2
    - F15
        - sample1
        - sample2
        - ...
        - sampleN
            - LEFT
                - *.TIF
            - RIGHT
                - *.TIF
```

## 3. run Calibration
```
sh ./scripts/intrinsic_calib_1.5.sh
sh ./scripts/psf_calib_1.5.sh
```

## 4. See log directory to find results