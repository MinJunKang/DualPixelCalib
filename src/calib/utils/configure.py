
import pdb
import json
import torch
import numpy as np
from src.calib.utils.file_manager import error_handler, option_check


class obj(object):
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [obj(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, obj(value) if isinstance(value, dict) else value)


class Configuration(object):
    
    def __init__(self, args):
        self.config = dict()

        # config from argparse
        self.config['calibname'] = args.calibname
        self.config['config'] = args.config
        self.config['model'] = args.model
        self.config['mode'] = args.mode
        self.config['verbose'] = True if args.no_verbose else False
        self.config['ngpus'] = torch.cuda.device_count()
        self.config['load_ckpt_name'] = args.load_ckpt_name

        # learning setting
        self.config['optim'] = 'adamw'  # [adam, adamw, radam]
        self.config['accelerator'] = 'dp'  # DP or DDP
        self.config['precision'] = 32  # 32 bit or 16 bit
        self.config['record_epoch'] = 5

        # read model's config
        with open('src/calib/model/%s/%s.json' % (args.model, args.config)) as json_file:
            json_data = json.load(json_file)
        args_data = json_data
        self.config.update(args_data)

        # read data's config
        data_info = np.load('dataset/%s/general_info.npy' % args.calibname, allow_pickle=True).item()
        self.config.update(data_info)

        # additional setting
        self.config['num_workers_img'] = 16
        self.config['num_workers'] = self.config['batch_size']
        self.config['sync_batch'] = True if self.config['accelerator'] is 'ddp' else False

        # check setting
        option_check(args.mode, ['train', 'eval'])
        option_check(self.config['purpose'], ['calib'])
        option_check(self.config['sensor_type'], ['dslr', 'phone', 'mixed'])
        option_check(self.config['device'], ['dslr', 'phone'])
        if self.config['sensor_type'] != 'mixed':
            error_handler(self.config['device'] == self.config['sensor_type'], 
                          'device not matched!', __name__, True)

    def update(self, new_config):
        if new_config is not None:
            self.config.update(new_config)
            
    def get_config(self):
        return obj(self.config)
