import os
import numpy as np 
import random
import torch.nn as nn
import math 
import torch
import json
from logging import getLogger
from time import perf_counter
import pandas as pd
from ignite.engine.engine import Engine, Events
from ignite.metrics import Average
import torch.nn.functional as F
import torch.optim as optim
from ignite.contrib.handlers import PiecewiseLinear, ParamGroupScheduler


def init_weights(module):    
    if isinstance(module, nn.Conv2d):    
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate(model, valid_loader):
    model.eval()
    valid_loss = []
    with torch.no_grad():
        for batch_i, (imgs, targets) in enumerate(valid_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            output = model(imgs)
            loss = criterion(output,targets)
            valid_loss.append(loss.cpu().numpy())
    return np.mean(valid_loss)

def get_lr_scheduler(optimizer, lr_max_value, lr_max_value_epoch, num_epochs, epoch_length):
    milestones_values = [
        (0, 0.00001), 
        (epoch_length * lr_max_value_epoch, lr_max_value), 
        (epoch_length * num_epochs - 1, 0.00001)
    ]
    lr_scheduler1 = PiecewiseLinear(optimizer, "lr", milestones_values=milestones_values,param_group_index=0)

    milestones_values = [
        (0, 0.00002), 
        (epoch_length * lr_max_value_epoch, lr_max_value * 2),
        (epoch_length * lr_max_value_epoch  + 5, lr_max_value),
        (epoch_length * num_epochs - 1, 0.00002)
    ]
    lr_scheduler2 = PiecewiseLinear(optimizer, "lr", milestones_values=milestones_values,param_group_index=1)

    lr_scheduler = ParamGroupScheduler(
        [lr_scheduler1, lr_scheduler2],
        ["lr scheduler (non-biases)", "lr scheduler (biases)"]
    )
    
    return lr_scheduler

def get_optimizer(model,lr, momentum, weight_decay, nesterov):
    biases = [p for n, p in model.named_parameters() if "bias" in n]
    others = [p for n, p in model.named_parameters() if "bias" not in n]
    return optim.SGD(
        [{"params": others, "lr": lr, "weight_decay": weight_decay}, 
         {"params": biases, "lr": lr, "weight_decay": weight_decay / 2}], 
        momentum=momentum, nesterov=nesterov
    )


def output_transform(output):
    _, pred_y, y = output
    return pred_y.cpu(), y.cpu() # must be cpu


def save_json(filepath, params):
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)


class DictOutputTransform:
    def __init__(self, key, index=0):
        self.key = key
        self.index = index

    def __call__(self, x):
        if self.index >= 0:
            x = x[self.index]
        return x[self.key]



def create_trainer(classifier, optimizer, device,w1,w2):
    classifier.to(device)

    def update_fn(engine, batch):
#         print(engine,batch)
        classifier.train()
        optimizer.zero_grad()
        # batch = [elem.to(device) for elem in batch]
        x, y = [elem.to(device) for elem in batch]
        x = x.to(device,dtype = torch.float)
        y = y.to(device,dtype = torch.float)

        preds = classifier(x)
        mse = F.mse_loss(preds, y)
        mae = F.l1_loss(preds,y)
        loss = (mse *w1 + mae*w2)/(w1+w2)
        metrics = {
                'loss': loss.item(),
                'mse':mse.item(),
                'mae':mae.item()

        }

        loss.backward()
        optimizer.step()
        return metrics,preds,y
    trainer = Engine(update_fn)
    for key in classifier.metrics_keys:
        Average(output_transform=DictOutputTransform(key)).attach(trainer, key)
    return trainer


def create_evaluator(classifier, device,w1,w2):
    classifier.to(device)

    def update_fn(engine, batch):
        classifier.eval()

        with torch.no_grad():
            # batch = [elem.to(device) for elem in batch]
            x, y = [elem.to(device) for elem in batch]
            x = x.to(device,dtype = torch.float)
            y = y.to(device,dtype = torch.float)
            preds = classifier(x)
            mse = F.mse_loss(preds, y)
            mae = F.l1_loss(preds,y)
            loss = (mse *w1 + mae*w2)/(w1+w2)
            metrics = {
                    'loss': loss.item(),
                    'mse':mse.item(),
                    'mae':mae.item()

            }
            return metrics,preds,y # return metric and pred, and y
    evaluator = Engine(update_fn)  

    for key in classifier.metrics_keys:
        Average(output_transform=DictOutputTransform(key)).attach(evaluator, key)
    return evaluator


class LogReport:
    def __init__(self, evaluator=None, dirpath=None, logger=None):
        self.evaluator = evaluator
        self.dirpath = str(dirpath) if dirpath is not None else None
        self.logger = logger or getLogger(__name__)

        self.reported_dict = {}  # To handle additional parameter to monitor
        self.history = []
        self.start_time = perf_counter()

    def report(self, key, value):
        self.reported_dict[key] = value

    def __call__(self, engine):
        elapsed_time = perf_counter() - self.start_time
        elem = {'epoch': engine.state.epoch,
                'iteration': engine.state.iteration}
        # print(engine.state.metrics.items())
        elem.update({'train/{}'.format(key): value for key, value in engine.state.metrics.items()})
        if self.evaluator is not None:
            elem.update({'valid/{}'.format(key): value
                         for key, value in self.evaluator.state.metrics.items()})
        elem.update(self.reported_dict)
        elem['elapsed_time'] = elapsed_time
        self.history.append(elem)
        if self.dirpath:
            save_json(os.path.join(self.dirpath, 'log.json'), self.history)
            self.get_dataframe().to_csv(os.path.join(self.dirpath, 'log.csv'), index=False)

        # --- print ---
        msg = ''
        for key, value in elem.items():
            # print("pair",key,value)
            if key in ['iteration']:
                # skip printing some parameters...
                continue
            elif isinstance(value, int):
                msg += '{} {}'.format(key,value)
            else:
                msg += '{} {}'.format(key,value)
#         self.logger.warning(msg)
        print(msg)

        # --- Reset ---
        self.reported_dict = {}

    def get_dataframe(self):
        df = pd.DataFrame(self.history)
        return df


class SpeedCheckHandler:
    def __init__(self, iteration_interval=10, logger=None):
        self.iteration_interval = iteration_interval
        self.logger = logger or getLogger(__name__)
        self.prev_time = perf_counter()

    def __call__(self, engine):
        if engine.state.iteration % self.iteration_interval == 0:
            cur_time = perf_counter()
            spd = self.iteration_interval / (cur_time - self.prev_time)
            self.logger.warning('{} iter/sec'.format(spd))
            # reset
            self.prev_time = cur_time

    def attach(self, engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)


class ModelSnapshotHandler:
    def __init__(self, model, filepath='model_{count:06}.pt',
                 interval=1, logger=None):
        self.model = model
        self.filepath= str(filepath)
        self.interval = interval
        self.logger = logger or getLogger(__name__)
        self.count = 0

    def __call__(self, engine):
        self.count += 1
        if self.count % self.interval == 0:
            filepath = self.filepath.format(count=self.count)
            torch.save(self.model.state_dict(), filepath)
            # self.logger.warning(f'save model to {filepath}...')

def init_gan_weights(module):
    assert isinstance(module, nn.Module)
    if hasattr(module, "weight") and module.weight is not None:
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if hasattr(module, "bias") and module.bias is not None:
        torch.nn.init.constant_(module.bias, 0.0)
    for c in module.children():
        init_weights(c)


def toggle_grad(model, on_or_off):
    # https://github.com/ajbrock/BigGAN-PyTorch/blob/master/utils.py#L674
    for param in model.parameters():
        param.requires_grad = on_or_off