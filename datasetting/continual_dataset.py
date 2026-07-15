import random
from datasetting.datasets import *
import numpy as np
from torch.utils.data import DataLoader
from utility import *


class DataManager(object):
    def __init__(self, args):
        self.args = args
        self.dataset_name = self.args.dataset
        self.get_data(args.shuffle)

    def get_joint_dataset(self, source, _indice=0):
        if source == "train":
            dataset = self.train_set
            index = self.train_index
            data_per_task = self.train_data_per_task
            if self.args.method == 'joint':
                indices = index[int(_indice * data_per_task): int((_indice+1) * data_per_task)] # joint1
            elif self.args.method == 'mtl':
                indices = index 
        elif source == "test":
            dataset = self.test_set
            index = self.test_index
            data_per_task = self.test_data_per_task
            indices = index
        else:
            raise ValueError("Unknown data source {}.".format(source))
        
        dataset._get_data(indices, self.args.cur_task)

        return dataset

    def get_dataset(self, source):
        if source == "train":
            dataset = self.train_set
            index = self.train_index
            data_per_task = self.train_data_per_task
            indices = index[int((self.args.cur_task - 1) * data_per_task) : int((self.args.cur_task)* data_per_task)]
        elif source == "test":
            dataset = self.test_set
            index = self.test_index
            data_per_task = self.test_data_per_task
            indices = index
        else:
            raise ValueError("Unknown data source {}.".format(source))
        
        dataset._get_data(indices, self.args.cur_task)

        return dataset


    def get_data(self, shuffle=False):
        if self.args.dataset == 'NYUv2':
            self.tasks_name = {1:"semantic", 2:'depth', 3:'normal'}
            self.train_set = NYUv2(root='/data/dataset/nyuv2', mode='train', augmentation=self.args.aug)
            self.test_set = NYUv2(root='/data/dataset/nyuv2', mode='test', augmentation=False)
            
        else:
            print("Dataset {} Not Support Currently".format(self.args.dataset))

        self.sequence = list(self.tasks_name.keys())
        self.task_num = len(list(self.tasks_name.keys())) // self.args.increments


        self.train_data_per_task = self.train_set.train_data_len // self.task_num
        self.test_data_per_task = self.test_set.test_data_len

        self.train_index = list(np.arange(self.train_set.train_data_len))
        self.test_index = list(np.arange(self.test_set.test_data_len))

        if shuffle:
            np.random.shuffle(self.train_index)
            np.random.shuffle(self.test_index)
        
        print_with_timestamp("The total number of task is", self.task_num, ', The current sequence of tasks are:', self.sequence)


        if self.args.print:
            print('The current index sequence is ', self.train_index)
                
    def task_sequence(self):
        return self.sequence

    def nb_tasks(self):
        return len(self.sequence)

    def task_type(self, cur_task):
        current_task_id = self.sequence[cur_task-1]
        a = self.tasks_name[current_task_id]
        return a

    def _task_names(self):
        m = list(self.tasks_name.values())
        return m