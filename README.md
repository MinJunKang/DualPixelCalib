
# User Guidance

## 1. Installation
```
git clone https://github.com/MinJunKang/DualPixelCalib.git -b baseline
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

## 3. Run Calibration
### **Stage1. Run Intrinsic Calibration**
```
sh ./scripts/calibration/intrinsic_calib_240912_F15.sh
```
### **Stage2. Run DP-PSF Calibration**
```
sh ./scripts/psf_calib_F15_A28_hash_240912.sh  # hash-grid (18GB), start from configs/model/psfvoxel_hash.yaml
sh ./scripts/psf_calib_F15_A28_rf_240912.sh  # tensorf (23GB), start from configs/model/psfvoxel_rf.yaml
```

## 4. See log directory to find results


## 5. evaluate model on ICCP20 Dataset
```
sh ./scripts/evaluate_F15_A28_hash_240912.sh  # test hash-grid
```