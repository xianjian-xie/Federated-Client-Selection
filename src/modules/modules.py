import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader
from utils import to_device, make_optimizer, collate


class Server:
    def __init__(self, model):
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        global_optimizer = make_optimizer(model, 'global')
        self.global_optimizer_state_dict = global_optimizer.state_dict()

    def distribute(self, client):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
        model.load_state_dict(self.model_state_dict)
        # print('model.state_dict().items()',model.state_dict().items())
        # for k, v in model.named_parameters():
            # print('k is', k)
            # print('v is', v.data)
            # break
        # print('model.state_dict().items()',model.named_parameters().keys())
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        # print('model_state_dict', model_state_dict.keys())
        # for k, v in model.state_dict().items():
        #     # print('k is', k)
        #     # print('v is', v)
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
                # print('model param', client[m].model_state_dict.keys())
        return

    def update(self, client):
        with torch.no_grad():
            valid_client = [client[i] for i in range(len(client)) if client[i].active]
            if len(valid_client) > 0:
                model = eval('models.{}()'.format(cfg['model_name']))
                model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
                model.load_state_dict(self.model_state_dict)
                global_optimizer = make_optimizer(model, 'global')
                global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                global_optimizer.zero_grad()
                weight = torch.ones(len(valid_client))
                weight = weight / weight.sum()
                for k, v in model.named_parameters():
                    print('k is', k)
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_client)):
                            tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                global_optimizer.step()
                self.global_optimizer_state_dict = global_optimizer.state_dict()
                self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            for i in range(len(client)):
                client[i].active = False
        return


class Client:
    def __init__(self, client_id, model, data_split):
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        self.active = False
        self.buffer = None
        self.good = True

    def train(self, dataset, lr, metric, logger):
        data_loader = make_data_loader({'train': dataset}, 'client')['train']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = make_optimizer(model, 'local')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        model.train(True)
        for epoch in range(1, cfg['local']['num_epochs'] + 1):
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return
