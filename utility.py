import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from datasetting.loss_functions import *
from datasetting.buffer import *
from argparse import ArgumentParser

mask=None


class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, tasks):
        self.database = 'PASCALContext'
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict

def get_single_task_meter(database, task):
    # {1:"semseg", 2:'human_parts', 3:'sal', 4:'normals', 5:'edge'}
    """ Retrieve a meter to measure the single-task performance """
    if task == 'semseg':
        from utils.eval_semseg import SemsegMeter
        return SemsegMeter(database)

    elif task == 'human_parts':
        from utils.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(database)

    elif task == 'normals':
        from utils.eval_normals import NormalsMeter
        return NormalsMeter()

    elif task == 'sal':
        from utils.eval_sal import SaliencyMeter
        return SaliencyMeter()

    elif task == 'edge': # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
        from utils.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=0.95)

    else:
        raise NotImplementedError

def get_output(output, task):
    output = output.permute(0, 2, 3, 1)
    
    if task == 'normals':
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
    
    elif task in {'semseg', 'human_parts'}:
        _, output = torch.max(output, dim=3)
    
    elif task in {'edge', 'sal'}:
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        pass
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output

def param_num(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_with_timestamp(f"Total number of trainable parameters: {trainable_params}")

def print_params(model):
    for name, param in model.named_parameters():
        print(param.data)  

def print_with_timestamp(message, *args):
    filename='trainer.py'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    additional_message = " ".join(map(str, args))
    print(f'{current_time} [{filename}] => {message} {additional_message}')


def model_fit(x_pred, x_output, task_type, dataset='NYUv2', mask=None):

    device = x_pred.device
    # if _idx is not None:
    #     x_output = x_output * _idx

    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type in ['semantic', 'segment_semantic', 'semseg', 'human_parts']:
        # semantic loss: depth-wise cross entropy
        _idx = 255 if dataset in ['Taskonomy', 'PASCAL'] else -1
        if dataset in ['PASCAL']:
            x_output = x_output[:, 0, :, :].long()
        if mask is not None:
            _compute = F.nll_loss(x_pred, x_output, ignore_index=_idx, reduction='none') # _compute[bs, W, H]
            _compute = _compute * mask
            loss = _compute.mean()
        else:    
            loss = F.nll_loss(x_pred, x_output, ignore_index=_idx)

    elif task_type in ['depth', 'depth_zbuffer', 'keypoints2d', 'edge_texture']:
        # depth loss: l1 norm
        if dataset in ['PASCAL']:
            binary_mask = (x_output != 255)
        if mask is not None:
            _compute = torch.abs(x_pred - x_output) * binary_mask
            _compute = _compute * mask
            loss = torch.sum(_compute) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        else:
            loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)


    elif task_type in ['normal', 'normals']:
        if dataset in ['PASCAL']:
            binary_mask = (x_output != 255)
        if mask is not None:
            _compute = (x_pred * x_output) * binary_mask
            _compute = _compute * mask
            loss = 1 - torch.sum(_compute) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        else:
        # normal loss: dot product
            loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    elif task_type in ['sal']:
        criterion = BalancedCrossEntropyLoss(size_average=True)
        loss = criterion(x_pred, x_output)
        
    else:
        RuntimeError("No available loss function for this dataset")


    return loss


def geo_mean(x):
    x_safe = x.clone()  
    x_safe[torch.isnan(x_safe)] = 0 

    x_safe = x_safe[x_safe != 0]

    if x_safe.numel() == 0: 
        return 0
    
    x_safe = torch.log(x_safe) 
    b = x_safe.mean()
    geo_mean = torch.exp(b)

    return geo_mean

def sober(image):
    image = image.unsqueeze(1) # [bs, 1, W, H]
    sobel_x = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()
    
    edge_x = F.conv2d(image, sobel_x, padding=1)
    edge_y = F.conv2d(image, sobel_y, padding=1)

    edges = torch.sqrt(edge_x**2 + edge_y**2)
    
    return edges.squeeze(1)


def num_dep_bar(x_pred, x_output, binary_mask, num_interval):
    loss = torch.abs(x_pred - x_output) * binary_mask 
    bar = num_interval

    min = x_pred < bar
    max = x_pred > bar

    a = loss[max].mean() if max.sum() > 0 else torch.tensor(0.0, device=loss.device)
    b = loss[min].mean() if min.sum() else torch.tensor(0.0, device=loss.device)

    # loss = torch.tensor([a, b]).mean()
    loss = (a + b).mean()
    return loss

def num_norm_bar(x_pred, x_output, binary_mask, num_interval):
    loss = (torch.mean(x_pred * x_output, dim=1).unsqueeze(1)) * binary_mask
    bar = num_interval
    min = x_pred.mean(dim=1) < bar
    max = x_pred.mean(dim=1) > bar

    a = 1 - loss[max.unsqueeze(1)].mean() if max.sum() > 0 else torch.tensor(1.0, device=loss.device)
    b = 1 - loss[min.unsqueeze(1)].mean() if min.sum() else torch.tensor(1.0, device=loss.device)

    # loss = torch.tensor([a, b]).mean()
    loss = (a + b).mean()
    return loss
    

def num_balance_seg(x_pred, x_output, loss, num_interval):
    num_classes = x_pred.shape[1]
    class_losses = torch.zeros(num_classes) 
    class_numbers = torch.zeros(num_classes) 
    class_conf = torch.zeros(num_classes)
    class_entropy = torch.zeros(num_classes)

    for i in range(num_classes):
        class_mask = (x_output == i)
        class_mask = class_mask
        class_pixel_count = class_mask.sum()
        if class_pixel_count:
            # class_losses[i] = (loss * class_mask).sum() / class_mask.sum()
            class_losses[i] = geo_mean(loss * class_mask)
            class_numbers[i] = class_mask.sum()
            m = torch.softmax(x_pred, dim=1)[:, i, :, :] # bs, W, H
            class_conf[i] = m.mean() # 1
            _entropy = m * class_mask
            entropy = -(_entropy * torch.log(_entropy + 1e-9))
            class_entropy[i] = geo_mean(entropy)
        else:
            class_losses[i] = 0  
            class_numbers[i] = 0  
            class_conf[i] = 0


    weights = torch.softmax( -1 * 1 * class_losses, dim=0).detach()

    weighted_class_losses = class_losses * weights
    loss = torch.nanmean(weighted_class_losses)

    return loss


def weight_fit(x_pred, x_output, task_type, args, dataset='NYUv2', mask=None):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type in ['semantic', 'segment_semantic', 'semseg', 'human_parts']:
    # semantic loss: depth-wise cross entropy
        _idx = 255 if dataset in ['Taskonomy', 'PASCAL'] else -1
        if dataset in ['PASCAL']:
            x_output = x_output[:, 0, :, :].long()
        if mask is not None:
            _compute = F.nll_loss(x_pred, x_output, ignore_index=_idx, reduction='none') # _compute[bs, W, H]
            _compute = _compute * mask
            loss = _compute.mean()
        else:    
            loss = F.nll_loss(x_pred, x_output, ignore_index=_idx, reduction='none')

            sober_mask = _edgeMask(loss, args.scalar_val, args.sel_max, task_type)
            loss_1 = loss * sober_mask
                
            if args.num_interval != 1:
                loss = num_balance_seg(x_pred, x_output, loss, args.num_interval)
            else:
                loss = F.nll_loss(x_pred, x_output, ignore_index=_idx, reduction='mean')

            loss = (loss + loss_1.mean()) / 2
                

    elif task_type in ['depth', 'depth_zbuffer', 'keypoints2d', 'edge_texture']:
        # depth loss: l1 norm
        if dataset in ['PASCAL']:
            binary_mask = (x_output != 255)
        if mask is not None:
            _compute = torch.abs(x_pred - x_output) * binary_mask
            _compute = _compute * mask
            loss = torch.sum(_compute) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        else:
            _compute = torch.abs(x_pred - x_output) * binary_mask
            non_zero_count = torch.nonzero(_compute, as_tuple=False).size(0)
            if non_zero_count == 0:
                loss_1 = torch.tensor(0)
            else:
                loss_1 = torch.sum(_compute) / non_zero_count

            sober_mask = _edgeMask(_compute.detach(), args.scalar_val, args.sel_max, task_type)
            mask_compute = _compute * sober_mask
            non_zero_count = torch.nonzero(mask_compute, as_tuple=False).size(0)
            if non_zero_count == 0:
                loss_2 = torch.tensor(0)
            else:
                loss_2 = torch.sum(mask_compute) / non_zero_count
            
         
            if args.num_interval == 1:
                loss = loss_1
            else:
                loss = num_dep_bar(x_pred, x_output, binary_mask, args.num_interval)
            
            loss = (loss +  loss_2) / 2
            

    elif task_type in ['normal', 'normals']:
        if dataset in ['PASCAL']:
            binary_mask = (x_output != 255)
        if mask is not None:
            _compute = (x_pred * x_output) * binary_mask
            _compute = _compute * mask
            loss = 1 - torch.sum(_compute) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        else:
            _compute = (x_pred * x_output) * binary_mask
            loss_1 = 1 - torch.mean(_compute, dim=1)

            sober_mask = _edgeMask(loss_1.detach(), args.scalar_val, args.sel_max, task_type)
            c = _compute * sober_mask.unsqueeze(1)
            non_zero_count = torch.nonzero(c, as_tuple=False).size(0)
            if non_zero_count == 0:
                loss_2 = torch.tensor(1)
            else:
                loss_2 = 1 - torch.sum(_compute) / non_zero_count

            if args.num_interval == 1:
                loss = torch.sum(loss_1) / torch.nonzero(binary_mask, as_tuple=False).size(0)
            else:
                loss = num_norm_bar(x_pred, x_output, binary_mask, args.num_interval)
            
            loss = (loss + loss_2) / 2

    elif task_type in ['sal']:
        criterion = BalancedCrossEntropyLoss(size_average=True)
        loss = criterion(x_pred, x_output)
    
    else:
        RuntimeError("No available loss function for this dataset")

    return loss

def _edgeMask(select, scalar_val, sel_max=True, task_type=None):
    sober_mask = sober(select.detach().squeeze(1))
    image_flat = sober_mask.view(select.shape[0], -1)

    min_vals = image_flat.min(dim=1, keepdim=True)[0]  # 每个批次的最小值
    max_vals = image_flat.max(dim=1, keepdim=True)[0] 
    image_normalized = (image_flat - min_vals) / (max_vals - min_vals)

    if sel_max == 1:
        mask = (image_normalized > scalar_val)
    elif sel_max == 0:
        mask = (image_normalized < scalar_val)
    else:
        mask = (image_normalized <= scalar_val) | (image_normalized >= (1 - scalar_val))

    sober_mask = mask.view(select.shape[0], select.shape[-2], select.shape[-1])
    # print('the number of nonzero pixels', torch.count_nonzero(sober_mask))


        
    return sober_mask


class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        with torch.no_grad():
            n = self.num_classes
            if self.mat is None:
                self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
            with torch.no_grad():
                k = (target >= 0) & (target < n)
                inds = n * target[k].to(torch.int64) + pred[k]
                self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        with torch.no_grad():
            h = self.mat.float()
            acc = torch.diag(h).sum() / h.sum()
            iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
            return torch.mean(iu).item(), acc.item()


def depth_error(x_pred, x_output, dataset='NYUv2'):
    with torch.no_grad():
        device = x_pred.device
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        if mask is not None:
            binary_mask *= (mask.int() == 1)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
               (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output, dataset='NYUv2'):
    with torch.no_grad():
        binary_mask = (torch.sum(x_output, dim=1) != 0)
        if mask is not None:
            binary_mask *= (mask[:,0,:,:].int() == 1)
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1))#.detach().cpu().numpy()
    #     error = np.degrees(error)
        error = torch.rad2deg(error)
        return torch.mean(error).item(), torch.median(error).item(), \
               torch.mean((error < 11.25)*1.0).item(), torch.mean((error < 22.5)*1.0).item(), \
               torch.mean((error < 30)*1.0).item()
    
   