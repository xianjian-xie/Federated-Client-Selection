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
from data import make_data_loader, separate_dataset
from utils import to_device, make_optimizer, collate


class Server:
    def __init__(self, model, data_split):
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        global_optimizer = make_optimizer(model, 'global')
        self.global_optimizer_state_dict = global_optimizer.state_dict()
        self.data_split = data_split

    def distribute(self, client):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
        model.load_state_dict(self.model_state_dict)
        # print('jiegou', vars(model))
        # print('model.state_dict().items() is',model.state_dict().items())
        # print('model key', dir(model))
        # for k, v in model.named_parameters():
        #     print('k is', k)
        #     print('v is', v)
        #     break
        # print('model.state_dict().items()',model.named_parameters().keys())
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        # print('model_state_dict', model_state_dict.keys())
        # for k, v in model.state_dict().items():
        #     print('k is', k)
        #     print('v is', v)
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
                # print('model param', client[m].model_state_dict.keys())
        return

    def select_client(self, client, dataset, optimizer, metric, logger, epoch, valid_client):
        dataset_server_validation = separate_dataset(dataset, self.data_split['train'])
        data_loader_server_validation = make_data_loader({'train': dataset_server_validation}, 'client')['train']

        for m in range(len(valid_client)):
            string_client_id = str(valid_client[m].client_id.item())
            
            model_validation = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model_validation.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            model_validation.load_state_dict(valid_client[m].model_state_dict, strict=False)
            # self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            # optimizer_validation = make_optimizer(model_validation, 'local')
            # optimizer_validation.load_state_dict(valid_client[m].optimizer_state_dict)
            # model_validation.train(True)
            
            # for epoch in range(1, cfg['local']['num_epochs'] + 1):
            for i, input in enumerate(data_loader_server_validation):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                # optimizer_validation.zero_grad()
                output = model_validation(input)
                # print('output', output)
                # output['loss'].backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                # optimizer_validation.step()
                evaluation = metric.evaluate(metric.metric_name['validation'], input, output)
                # print('eval validation:', evaluation)
                logger.append(evaluation, 'validation'+string_client_id, n=input_size)
            
            info = {'info': ['Model: {}'.format(cfg['model_tag'])
                            # 'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                            # 'Learning rate: {:.6f}'.format(lr),
                            # 'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
                            # 'Epoch Finished Time: {}'.format(epoch_finished_time),
                            # 'Experiment Finished Time: {}'.format(exp_finished_time)
                            ]}
            
            logger.append(info, 'validation'+string_client_id, mean=False)
            print(m, valid_client[m].client_id, logger.write('validation'+string_client_id, metric.metric_name['validation']))
            name = 'validation'+string_client_id+'/Accuracy'
            print('logger mean', logger.mean[name])
            if logger.mean[name] < 15:
                valid_client[m].good = False
                print('huaidan', valid_client[m].client_id)
        


    def update(self, client, dataset, optimizer, metric, logger, epoch):
        # with torch.no_grad():
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
            ###################################
            self.select_client(client, dataset, optimizer, metric, logger, epoch, valid_client)
            # dataset_server_validation = separate_dataset(dataset, self.data_split['train'])
            # data_loader_server_validation = make_data_loader({'train': dataset_server_validation}, 'client')['train']

            # for m in range(len(valid_client)):
            #     string_client_id = str(valid_client[m].client_id.item())
                
            #     model_validation = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            #     model_validation.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            #     model_validation.load_state_dict(valid_client[m].model_state_dict, strict=False)
            #     # self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            #     # optimizer_validation = make_optimizer(model_validation, 'local')
            #     # optimizer_validation.load_state_dict(valid_client[m].optimizer_state_dict)
            #     # model_validation.train(True)
                
            #     # for epoch in range(1, cfg['local']['num_epochs'] + 1):
            #     for i, input in enumerate(data_loader_server_validation):
            #         input = collate(input)
            #         input_size = input['data'].size(0)
            #         input = to_device(input, cfg['device'])
            #         # optimizer_validation.zero_grad()
            #         output = model_validation(input)
            #         # print('output', output)
            #         # output['loss'].backward()
            #         # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            #         # optimizer_validation.step()
            #         evaluation = metric.evaluate(metric.metric_name['validation'], input, output)
            #         print('eval validation:', evaluation)
            #         logger.append(evaluation, 'validation'+string_client_id, n=input_size)
                
            #     info = {'info': ['Model: {}'.format(cfg['model_tag'])
            #                     # 'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
            #                     # 'Learning rate: {:.6f}'.format(lr),
            #                     # 'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
            #                     # 'Epoch Finished Time: {}'.format(epoch_finished_time),
            #                     # 'Experiment Finished Time: {}'.format(exp_finished_time)
            #                     ]}
                
            #     logger.append(info, 'validation'+string_client_id, mean=False)
            #     print(m, valid_client[m].client_id, logger.write('validation'+string_client_id, metric.metric_name['validation']))
            ###################################
            for k, v in model.named_parameters():
                # print('k is', k)
                # print('v is', v.size())
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v = v.data.new_zeros(v.size())
                    for m in range(len(valid_client)):
                        # print('model state dict', valid_client[m].model_state_dict[k].size())
                        if valid_client[m].good == True:
                            print('you', valid_client[m].client_id)
                            tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        else:
                            continue
                    v.grad = (v.data - tmp_v).detach()
            global_optimizer.step()
            self.global_optimizer_state_dict = global_optimizer.state_dict()
            self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        for i in range(len(client)):
            client[i].active = False
            client[i].good = True
        return


class Client:
    def __init__(self, client_id, model, data_split):
        self.client_id = client_id
        # print('client id', type(client_id), client_id)
        self.data_split = data_split
        # data split是一个字典 {train: [0, 12, 23], test: [1,9]}
        # print('self.data_split',self.data_split)
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        self.active = False
        self.buffer = None
        self.good = True

    def train(self, dataset, lr, metric, logger):
        data_loader = make_data_loader({'train': dataset}, 'client')['train']
        # print('data loader client', data_loader, len(data_loader))
        # for i, input in enumerate(data_loader):
        #     print('loader', i)
            # break
        # dataloader的每一项是10个id, 10个data，10个target, 一共49个loader, len(data_loader)=50
        # dataset 有495个样本
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
                # print('input22', input['data'].size(), input['data'][0])  # input22 torch.Size([10, 3, 32, 32]) 3个通道
                # break
                input_size = input['data'].size(0)
                # print('input size', input_size)   # input_size 10
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                # print('output1', output)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                # print('evaluation22', evaluation) #evaluation22 {'Loss': 2.296760320663452, 'Accuracy': 20.0}
                logger.append(evaluation, 'train', n=input_size)
        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return
