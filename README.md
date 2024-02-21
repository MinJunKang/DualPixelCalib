<<<<<<< HEAD

# Download and Install

## 1. Git clone
```
git clone --recursive https://github.com/MinJunKang/DualPixelCalib.git
```

## 2. Download calib dataset and put them under dataset directory.
```
- boards
- src
  - calib
  - endtask
- dataset
  - put your own dataset here !! [dataset name]
  main_calibrate.py
  main_endtask.py
  README.md
  requirements.txt
```

## 3. Download checkpoint from [link](https://drive.google.com/open?id=1MnoyTZgHgG7vwZYuloLeQgegwa6O9A7O&authuser=codeslake%40gmail.com&usp=drive_fs) and put them into src/calib/extern/IFAN/ckpt

## 4. pip install -r requirements.txt (Docker will be provided later)

## 5. calibration command
```
python main_calibrate.py --calibname [dataset name] --model vox_mixed --config config_nomlp
```
=======

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
>>>>>>> 7b3e42110a572a3910427a3e4df7fba191c3dcb4
