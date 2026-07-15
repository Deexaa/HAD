from utility import *
from datasetting.continual_dataset import DataManager
from torch.utils.data import DataLoader

import numpy as np
import random, torch

from methods import get_model

def lhl_train(args):
    data_manager = DataManager(args)
    args.cur_task += 1

    for key, value in vars(args).items():
        print(f'{key}: {value}')
    
    if data_manager.task_num < args.increments:
        raise NotImplementedError('Current LHL setting is not suitable')

    # current_metrics_per_task = []

    model = get_model(args)
    
    if args.method != 'mtl':
        print_with_timestamp('Starting new run')
        
        for task in range(data_manager.task_num):
            model.incremental_train(data_manager)
            model.after_task()
            model.eval_task()
    else:
        model.incremental_train(data_manager)
        model.after_task()
        model.eval_task()
        
    
    print_with_timestamp('======== Finishing Run ========')



