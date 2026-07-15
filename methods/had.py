from continual_model import build_model
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from utility import *
from continual_model import *
              

class HAD(BaseLearner):

    NAME = 'had'

    def __init__(self, args):
        super().__init__(args)
        self._network = build_model(args).cuda()
        self.soft = torch.nn.Softmax(dim=1)
        self.args = self.add_args_had(args)
        

    @staticmethod
    def add_args_had(args):
        parser = ArgumentParser()
        parser.add_argument('--new', default=3, type=float)
        parser.add_argument('--scalar_val', default=0.9, type=float)
        parser.add_argument('--num_interval', default=0.5)
        parser.add_argument('--td_ba', default=1, type=float)
        parser.add_argument('--sel_max', default=1)
        new_args = parser.parse_args(namespace=args)
        return new_args


    def incremental_train(self, data_manager):
        self.task_type = data_manager.task_type(self.args.cur_task)
        self.competence.append(self.task_type)
        
        print_with_timestamp("Starting the {} HCL Task, ".format(self.args.cur_task), "type =", self.task_type)
        
        param_num(self._network)
        self._network.update_decoder(self.task_type)
        print_with_timestamp('Update decoder')
        param_num(self._network)

        train_dataset = data_manager.get_dataset(source="train")
        self.train_loader = DataLoader(train_dataset, batch_size=int(self.args.batch_size), shuffle=False, drop_last=True)
        test_dataset = data_manager.get_dataset(source="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        self._train()


    def _train(self):
        self._network.cuda()
        optimizer = optim.Adam(self._network.parameters(), lr=self.args.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        self._new_train(self.train_loader, self.test_loader, optimizer, scheduler)

    def _new_train(self, train_loader, test_loader, optimizer, scheduler):
        for _, epoch in enumerate(range(int(self.args.epoch))):
            self._network.train()
            train_losses = 0
            cl_losses = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.cuda()
                img_size = inputs.size()[-2:]
                targets = targets[self.task_type]
                targets = self.label_fit(self.task_type, targets)
                
                cl_loss = 0
                
                features  = self._network.get_features(inputs)
                train_pred = self._network.get_decoder(features, [self.task_type], img_size)[self.task_type]
                train_loss = model_fit(train_pred, targets, self.task_type, self.args.dataset)

                if self.args.cur_task > 1:
                    new_loss = 0
                    for _task in self.competence[:-1]:
                        with torch.no_grad(): 
                            replay_target = self.checkpoint(inputs, [_task])[_task]

                        if _task in ['semantic', 'segment_semantic', 'semseg', 'human_parts']:
                            replay_target = torch.argmax(replay_target, dim=1)
                            if self.args.dataset == 'PASCAL':
                                replay_target = replay_target.unsqueeze(1)

                        replay_pred = self._network.get_decoder(features, [_task], img_size)[_task]

                        _loss = weight_fit(replay_pred, replay_target, _task, self.args, self.args.dataset)
                        _loss = _loss.cuda()
                        new_loss += _loss 
                        
                    cl_loss = self.args.new * new_loss
                    cl_losses += cl_loss
                    
                train_losses += train_loss
                loss = train_loss + cl_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                print("Task {}, Epoch {}/{} => Loss {:.4f}, CL_Loss {:.4f}".format(
                    self.args.cur_task,
                    epoch + 1,
                    self.args.epoch,
                    train_losses / len(train_loader),
                    cl_losses / len(train_loader)
                ))


    def after_task(self):
        self.args.cur_task += 1
        self.checkpoint = deepcopy(self._network)
        self.checkpoint.eval()


