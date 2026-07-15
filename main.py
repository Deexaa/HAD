import argparse
import torch, os
import random
from utility import *
from trainer import lhl_train
import numpy as np

def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED']= str(seed)


def parse_args():
    parser = argparse.ArgumentParser(description= 'LHL')
    parser.add_argument('--dataset', type=str, default='NYUv2')
    parser.add_argument('--aug', action='store_true', default=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--increments', default=1, type=float)
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--cur_task', default=0, type=int)
    parser.add_argument('--shuffle', default=False, type=bool)
    parser.add_argument('--print', default=False, type=float)
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--method', default='had', type=str)


    return parser.parse_args()

def main(**kwargs):
    args = parse_args()
    
    for k, v in kwargs.items():
        setattr(args, k, v)


    fix_seed(args.seed)
    lhl_train(args)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    main()
