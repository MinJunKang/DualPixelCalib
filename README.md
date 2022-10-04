
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
python main_calibrate.py --calibname [dataset name] --model vox_mixed --config config_nomlp
