import torch, sys
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
from model.resnet_dilated import ResnetDilated
from model.aspp import DeepLabHead

from utils.basemodel import BaseModel
from utility import *


from torch.utils.data import DataLoader
from datasetting.custom_collate import collate_mil
import matplotlib.cm as cm

from copy import deepcopy
import torch

import matplotlib.pyplot as plt
import time

class BaseLearner(object):
    def __init__(self, args):
        self.competence = []
        self._known_tasks = 0
        self.args=args
        self.tasks_increments=[]
        self._network = None

    def after_task(self):
        self.args.cur_task += 1


    def add_rehearsal_args(self, args):
        parser = ArgumentParser()
        parser.add_argument('--buffer_size', default=50, type=int,
                            help='Penalty weight.')
        parser.add_argument('--minibatchsize', default=8, type=int,
                            help='Penalty weight.')
        new_args = parser.parse_args(namespace=args)
        return new_args


    def label_fit(self, task_type, targets):
        if task_type in ['semantic', 'segment_semantic']:
            targets = targets.long().cuda()
        else:
            targets = targets.cuda()

        return targets


    def get_loss(self, task):
        """ Return loss function for a specific task """

        if task == 'edge':
            criterion = BalancedCrossEntropyLoss(size_average=True, pos_weight=0.95)

        elif task == 'semseg' or task == 'human_parts':
            criterion = SoftMaxwithLoss()

        elif task == 'normals':
            criterion = NormalsLoss(normalize=True, size_average=True, norm=1)

        elif task == 'sal':
            criterion = BalancedCrossEntropyLoss(size_average=True)

        elif task == 'depth':
            criterion = DepthLoss('l1')

        else:
            raise NotImplementedError('Undefined Loss: Choose a task among '
                                    'edge, semseg, human_parts, sal, depth, or normals')

        return criterion


    
    def eval_task(self):
        loader = self.test_loader
        self._network.eval()
        if 'semantic' in self.competence or 'segment_semantic' in self.competence:
            conf_mat = ConfMatrix(self._network.class_nb)

        with torch.no_grad():  # operations inside don't track history
            val_dataset = iter(loader)
            val_batch = len(loader)

            if self.args.dataset in ['NYUv2', 'CityScape']:
                avg_cost = torch.zeros([9,1])
                cost_iter = torch.zeros([9,1])

                for k in range(val_batch):
                    val_data, targets = next(val_dataset)
                    targets = dict(filter(lambda item: item[0] in self.competence, targets.items()))
                    if 'semantic' in targets:
                        targets['semantic'] = targets['semantic'].long()
                    val_data= val_data.cuda()
                    for key in targets:
                        targets[key] = targets[key].cuda()

                    val_pred = self._network(val_data, self.competence)

                    for _type in self.competence: # for all competence
                        if _type in ['semantic', 'segment_semantic']:
                            conf_mat.update(val_pred[_type].argmax(1).flatten(), targets[_type].flatten())
                            if 0:
                                self.save_visual(val_pred[_type], val_data, targets[_type])
                            
                        if _type in ['depth']:
                            cost_iter[2], cost_iter[3] = depth_error(val_pred[_type], targets[_type])
                            if 0:
                                self.save_regression(val_pred[_type], val_data, targets[_type], type=_type)
                            if 0:
                                self.predict_plot(val_pred[_type], val_data, targets[_type], type=_type)
                        if _type in ['normal']:
                            cost_iter[4], cost_iter[5], cost_iter[6], cost_iter[7], cost_iter[8] = normal_error(val_pred[_type], targets[_type])
                            if 0:
                                self.save_regression(val_pred[_type], val_data, targets[_type], type=_type)
                            if 0:
                                self.predict_plot(val_pred[_type], val_data, targets[_type], type=_type)
                    
                    avg_cost += cost_iter / val_batch
                # compute mIoU and acc
                if 'semantic' in self.competence or 'segment_semantic' in self.competence:
                    avg_cost[0], avg_cost[1] = conf_mat.get_metrics()

                print_with_timestamp("semantic results are {} depth results are {}, normal results are {}".format(
                    ["{:.4f}".format(avg_cost[0].item()), "{:.4f}".format(avg_cost[1].item())],
                    ["{:.4f}".format(avg_cost[2].item()), "{:.4f}".format(avg_cost[3].item())],
                    ["{:.4f}".format(avg_cost[4].item()), "{:.4f}".format(avg_cost[5].item()), "{:.4f}".format(avg_cost[6].item()), "{:.4f}".format(avg_cost[7].item()), "{:.4f}".format(avg_cost[8].item())]
                ))

            elif self.args.dataset == 'PASCAL':
                performance_meter = PerformanceMeter(self.competence)
                for k in range(val_batch):
                    val_data, targets = next(val_dataset)
                    targets = dict(filter(lambda item: item[0] in self.competence, targets.items()))
                    if 'semantic' in targets:
                        targets['semantic'] = targets['semantic'].long()
                    val_data= val_data.cuda()
                    for key in targets:
                        targets[key] = targets[key].cuda()

                    val_pred = self._network(val_data, self.competence)

                    performance_meter.update({t: get_output(val_pred[t], t) for t in self.competence}, 
                                 {t: targets[t] for t in self.competence})
                eval_results_test = performance_meter.get_score(verbose=False)
                print_with_timestamp(eval_results_test)


def build_model(args):
    model = DeepLabv3(args)
    return model

class DeepLabv3(BaseModel):
    def __init__(self, args):
        
        ch = [256, 512, 1024, 2048]
        
        if args.dataset == 'NYUv2':
            self.class_nb = 13
            self.tasks = ['semantic', 'depth', 'normal']
            self.num_out_channels = {'semantic': 13, 'depth': 1, 'normal': 3}
        elif args.dataset == 'CityScape':
            self.class_nb = 7
            self.tasks = ['semantic', 'depth']
            self.num_out_channels = {'semantic': 7, 'depth': 1}
        elif args.dataset == 'PASCAL':
            self.class_nb = 21
            self.tasks =  ['semseg', 'human_parts', 'sal', 'normals', 'edge']
            self.num_out_channels = {'semseg': 21, 'human_parts': 7, 'sal': 1,
                                     'normals': 3, 'edge': 1}
        else:
            raise('No support {} dataset'.format(args.dataset))
        self.task_num = len(self.tasks)
        
        super(DeepLabv3, self).__init__(task_num=self.task_num)
        
        self.backbone = ResnetDilated(resnet.__dict__[args.model](pretrained=True)) # Change Backbone to another
        m = list(self.num_out_channels.values())[args.cur_task-1]
        self.decoders = {}
        self.args = args
        
        

    def update_decoder(self, task_type):
        if isinstance(task_type, str):
            task_type = [task_type] 
        for _task_type in task_type:
            new_decoder = DeepLabHead(512 if self.args.model=='resnet18' else 2048, self.num_out_channels[_task_type]).cuda()
            self.decoders[_task_type] = new_decoder
            self.add_module(f"decoder_{_task_type}", new_decoder)

    def print_params(self):
        for param in self.backbone.parameters():
            print(param.data)

    def get_feat_size(self):
        return 512
        


    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                    + 100 * output_size + output_size)
        """
        params = []
        for pp in list(self.backbone.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                   + 100 * output_size + output_size)
        """
        grads = []
        for pp in list(self.backbone.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def get_features(self, x):
        img_size  = x.size()[-2:]
        x = self.backbone(x)
        return x

    def get_decoder(self, x, task_type, img_size):
        out_dict = {}
        for _task_type in task_type:
            out = self.decoders[_task_type](x) 
            out = F.interpolate(out, img_size, mode='bilinear', align_corners=True)
            if _task_type in ['semantic', 'segment_semantic']:
                out = F.log_softmax(out, dim=1)
            elif _task_type in ['normal']:
                out = out / torch.norm(out, p=2, dim=1, keepdim=True)
            out_dict[_task_type] = out
        return out_dict

        
    def forward(self, x, task_type):
        img_size  = x.size()[-2:]
        x = self.backbone(x)
        self.rep = x
        if self.rep_detach:
            for tn in range(self.task_num):
                self.rep_i[tn] = self.rep.detach().clone()
                self.rep_i[tn].requires_grad = True

        out_dict = {}
        if len(task_type) != len(x):
            for _task_type in task_type:
                out = self.decoders[_task_type](x) 
                out = F.interpolate(out, img_size, mode='bilinear', align_corners=True)
                if _task_type in ['semantic', 'segment_semantic', 'semseg', 'human_parts']:
                    out = F.log_softmax(out, dim=1)
                elif _task_type in ['normal', 'normals']:
                    out = out / torch.norm(out, p=2, dim=1, keepdim=True)
            
                out_dict[_task_type] = out
        elif len(task_type) == len(x):
            self.eval()
            unique_tasks = list(set(task_type))
            for _task in unique_tasks:
                indices = [u for u in range(len(x)) if task_type[u] == _task]
                out = self.decoders[_task](x[indices]) 
                out = F.interpolate(out, img_size, mode='bilinear', align_corners=True)
                if _task in ['semantic', 'segment_semantic', 'semseg', 'human_parts']:
                    out = F.log_softmax(out, dim=1)
                elif _task in ['normal', 'normals']:
                    out = out / torch.norm(out, p=2, dim=1, keepdim=True)
                
                out_dict[_task] = out

            self.train()

        return out_dict
    

    def get_share_params(self):
        return self.backbone.parameters()



