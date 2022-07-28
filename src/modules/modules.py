import copy
import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sys
import time
import torch
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader, separate_dataset
from utils import to_device, make_optimizer, collate, write_log
import os
from printlog import make_print_to_file
from data import add_noise_to_model 
# from printlog import Logger
np.set_printoptions(linewidth=400)

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
            # print('k is', k)
            # print('v is', v.size())
            # print('v is', v)
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
                # print('model param', client[m].model_state_dict.keys())
        return

    def find_outlier(self, relation_list, param, tag):
        outlier_idx = []
        q1 = np.quantile(relation_list, 0.25)
        q3 = np.quantile(relation_list, 0.75)
        iqr = q3 - q1
        low, up = q1 - param*iqr, q3 + param*iqr
        for i in range(len(relation_list)):
            if tag == 'high':
                # print('jin high')
                if relation_list[i] > up:
                    outlier_idx.append(i)
            else:
                # print('jin low')
                if relation_list[i] <low:
                    outlier_idx.append(i)
        print('suoyou', q1,q3,iqr,low,up,outlier_idx)
        return outlier_idx


    def find_outlier1(self, relation_list):
        X = np.array(relation_list).reshape((-1,1))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        label = kmeans.labels_
        print('label', label)
        print('center', kmeans.cluster_centers_, kmeans.cluster_centers_.shape)
        if kmeans.cluster_centers_[0,0] < kmeans.cluster_centers_[1,0]:
            outlier_idx = np.where(label==1)[0]
        else:
            outlier_idx = np.where(label==0)[0]
        outlier_idx = outlier_idx.tolist()
        return outlier_idx

    def find_outlier2(self, relation_list):
        X = np.array(relation_list).reshape((-1,1))
        gm = GaussianMixture(n_components=2, random_state=0).fit(X)
        label = gm.predict(X)
        print('label', label)
        print('center', gm.means_, gm.means_.shape)
        if gm.means_[0,0] < gm.means_[1,0]:
            outlier_idx = np.where(label==1)[0]
        else:
            outlier_idx = np.where(label==0)[0]
        outlier_idx = outlier_idx.tolist()
        return outlier_idx



    def select_client_all_correct(self, client, dataset, optimizer, metric, logger, epoch):
        valid_client = [client[i] for i in range(len(client)) if client[i].active]
        for i in range(len(valid_client)):
            if valid_client[i].set_malicious == True:
                valid_client[i].good = False
                print('huaidan', epoch, valid_client[i].client_id)
        true_positive = 0
        false_negative = 0
        false_positive = 0
        true_negative = 0
        real_list = []
        prediction_list = []
        for i in range(len(valid_client)):
            if valid_client[i].set_malicious == True and valid_client[i].good == False:
                true_positive += 1
                real_list.append(1) # 坏人记为1,好人记为0
                prediction_list.append(1)
            elif valid_client[i].set_malicious == True and valid_client[i].good == True:
                false_negative += 1
                real_list.append(1)
                prediction_list.append(0)
            elif valid_client[i].set_malicious == False and valid_client[i].good == False:
                false_positive += 1
                real_list.append(0)
                prediction_list.append(1)
            else:
                true_negative += 1
                real_list.append(0)
                prediction_list.append(0)
        print('true positive is', epoch, true_positive, false_negative, false_positive, true_negative)
        print('real list is', epoch, real_list, prediction_list)


    def select_client_cluster_density(self, client, dataset, optimizer, metric, logger, epoch):
        valid_client = [client[i] for i in range(len(client)) if client[i].active]

        dataset_server_validation = separate_dataset(dataset, self.data_split['train'])
        data_loader_server_validation = make_data_loader({'train': dataset_server_validation}, 'client')['train']
        relation = []
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
            # print(m, valid_client[m].client_id, logger.write('validation'+string_client_id, metric.metric_name['validation']))
            name = 'validation'+string_client_id+'/Accuracy'
            print('logger mean', 'epoch', epoch, valid_client[m].client_id, logger.mean[name])
            relation.append(logger.mean[name])
        outlier_idx = self.find_outlier(relation, 1.5, 'low')
        print('outlier_idx is', epoch, outlier_idx)
        for i in range(len(outlier_idx)):
            valid_client[outlier_idx[i]].good = False
            print('huaidan', epoch, valid_client[outlier_idx[i]].client_id)
        true_positive = 0
        false_negative = 0
        false_positive = 0
        true_negative = 0
        real_list = []
        prediction_list = []
        for i in range(len(valid_client)):
            if valid_client[i].set_malicious == True and valid_client[i].good == False:
                true_positive += 1
                real_list.append(1) # 坏人记为1,好人记为0
                prediction_list.append(1)
            elif valid_client[i].set_malicious == True and valid_client[i].good == True:
                false_negative += 1
                real_list.append(1)
                prediction_list.append(0)
            elif valid_client[i].set_malicious == False and valid_client[i].good == False:
                false_positive += 1
                real_list.append(0)
                prediction_list.append(1)
            else:
                true_negative += 1
                real_list.append(0)
                prediction_list.append(0)
        print('true positive is', epoch, true_positive, false_negative, false_positive, true_negative)
        print('real list is', epoch, real_list, prediction_list)
            
        return

        

    def select_client_validation_set(self, client, dataset, optimizer, metric, logger, epoch):
        valid_client = [client[i] for i in range(len(client)) if client[i].active]

        dataset_server_validation = separate_dataset(dataset, self.data_split['train'])
        data_loader_server_validation = make_data_loader({'train': dataset_server_validation}, 'client')['train']
        relation = []
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
            # print(m, valid_client[m].client_id, logger.write('validation'+string_client_id, metric.metric_name['validation']))
            name = 'validation'+string_client_id+'/Accuracy'
            print('logger mean', 'epoch', epoch, valid_client[m].client_id, logger.mean[name])
            relation.append(logger.mean[name])
        outlier_idx = self.find_outlier(relation, 1.5, 'low')
        print('outlier_idx is', epoch, outlier_idx)
        for i in range(len(outlier_idx)):
            valid_client[outlier_idx[i]].good = False
            print('huaidan', epoch, valid_client[outlier_idx[i]].client_id)
        true_positive = 0
        false_negative = 0
        false_positive = 0
        true_negative = 0
        real_list = []
        prediction_list = []
        for i in range(len(valid_client)):
            if valid_client[i].set_malicious == True and valid_client[i].good == False:
                true_positive += 1
                real_list.append(1) # 坏人记为1,好人记为0
                prediction_list.append(1)
            elif valid_client[i].set_malicious == True and valid_client[i].good == True:
                false_negative += 1
                real_list.append(1)
                prediction_list.append(0)
            elif valid_client[i].set_malicious == False and valid_client[i].good == False:
                false_positive += 1
                real_list.append(0)
                prediction_list.append(1)
            else:
                true_negative += 1
                real_list.append(0)
                prediction_list.append(0)
        print('true positive is', epoch, true_positive, false_negative, false_positive, true_negative)
        print('real list is', epoch, real_list, prediction_list)
            
        return
        
    def select_client_cosine(self, client, dataset, optimizer, metric, logger, epoch):

        valid_client = [client[i] for i in range(len(client)) if client[i].active]

        dataset_server_validation = separate_dataset(dataset, self.data_split['train'])
        data_loader_server_validation = make_data_loader({'train': dataset_server_validation}, 'client')['train']

        if cfg['diff_option'] == 'diff':
            model_0 = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model_0.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            model_0.load_state_dict(self.model_state_dict, strict=False)
            model_0_dict = {k: v.cpu() for k, v in model_0.state_dict().items()}


        model_server = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model_server.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
        model_server.load_state_dict(self.model_state_dict, strict=False)
        lr = optimizer.param_groups[0]['lr']
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer_server = make_optimizer(model_server, 'local')
        optimizer_server.load_state_dict(self.optimizer_state_dict)
        model_server.train(True)

        for epoch1 in range(1, cfg['local']['num_epochs'] + 1):
            for i, input in enumerate(data_loader_server_validation):
                input = collate(input)
                # print('input22', input['data'].size(), input['data'][0])  # input22 torch.Size([10, 3, 32, 32]) 3个通道
                # break
                input_size = input['data'].size(0)
                # print('input size', input_size)   # input_size 10
                input = to_device(input, cfg['device'])
                optimizer_server.zero_grad()
                output = model_server(input)
                # print('output1', output)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model_server.parameters(), 1)
                optimizer_server.step()
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                # print('evaluation22', evaluation) #evaluation22 {'Loss': 2.296760320663452, 'Accuracy': 20.0}
                logger.append(evaluation, 'validation2', n=input_size)
        # self.optimizer_state_dict = optimizer_server.state_dict()
        # self.model_state_dict = {k: v.cpu() for k, v in model_server.state_dict().items()}
        model_server_param_vector = torch.zeros([1,1])
        model_server_dict = {k: v.cpu() for k, v in model_server.state_dict().items()}
        for k, v in model_server_dict.items():
        # for i in list(model_server.state_dict().values()): 
            if cfg['diff_option'] == 'diff':
                single_server_param_vector = torch.reshape(torch.sub(model_server_dict[k],model_0_dict[k]), (-1,1))
            elif cfg['diff_option'] == 'no-diff':
                single_server_param_vector = torch.reshape(model_server_dict[k], (-1,1))
            else:
                print('diff doumeijin')
            model_server_param_vector = torch.cat((model_server_param_vector, single_server_param_vector), 0)
        # print('model server vector', model_server_param_vector, model_server_param_vector.size())


        relation = []

        for m in range(len(valid_client)):
            string_client_id = str(valid_client[m].client_id.item())

            model_m = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model_m.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            model_m.load_state_dict(valid_client[m].model_state_dict, strict=False)

            model_m_param_vector = torch.zeros([1,1])
            model_m_dict = {k: v.cpu() for k, v in model_m.state_dict().items()}

            for k, v in model_m_dict.items():
                if cfg['diff_option'] == 'diff':
                    single_m_param_vector = torch.reshape(torch.sub(model_m_dict[k], model_0_dict[k]), (-1,1))
                elif cfg['diff_option'] == 'no-diff':
                    single_m_param_vector = torch.reshape(model_m_dict[k], (-1,1))
                else:
                    print('diff doumeijin2')
                model_m_param_vector = torch.cat((model_m_param_vector, single_m_param_vector), 0)

            # print('model m vector', model_m_param_vector,  model_m_param_vector.size())
            output1 = F.cosine_similarity(model_server_param_vector, model_m_param_vector, dim = 0).item()
            # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            # output = cos(model_server_param_vector, model_m_param_vector)
            # distance = torch.cdist(model_server_param_vector, model_m_param_vector, p=2)
            print('output is', valid_client[m].client_id, output1)
            # print('distance is', valid_client[m].client_id, distance)
            relation.append(output1)
        print('relation is', relation)
        ###
        outlier_idx = self.find_outlier(relation, 1.5, 'low')
        print('outlier_idx is', epoch, outlier_idx)
        for i in range(len(outlier_idx)):
            valid_client[outlier_idx[i]].good = False
            print('huaidan', epoch, valid_client[outlier_idx[i]].client_id)
        true_positive = 0
        false_negative = 0
        false_positive = 0
        true_negative = 0
        real_list = []
        prediction_list = []
        for i in range(len(valid_client)):
            if valid_client[i].set_malicious == True and valid_client[i].good == False:
                true_positive += 1
                real_list.append(1) # 坏人记为1,好人记为0
                prediction_list.append(1)
            elif valid_client[i].set_malicious == True and valid_client[i].good == True:
                false_negative += 1
                real_list.append(1)
                prediction_list.append(0)
            elif valid_client[i].set_malicious == False and valid_client[i].good == False:
                false_positive += 1
                real_list.append(0)
                prediction_list.append(1)
            else:
                true_negative += 1
                real_list.append(0)
                prediction_list.append(0)
        print('true positive is', epoch, true_positive, false_negative, false_positive, true_negative)
        print('real list is', epoch, real_list, prediction_list)


        ###
        # min_value = min(relation)
        # min_index = relation.index(min_value)
        # # print('min index is', min_index)
        # valid_client[min_index].good = False
        # print('huaidan', valid_client[min_index].client_id)

    def select_client_zero_one(self, client, dataset, optimizer, metric, logger, epoch):
        # root = os.path.abspath(os.path.join(__file__ ,"../..")) # src文件夹下
        # print('root is', root)
        # print('epoch is', epoch)

        valid_client = [client[i] for i in range(len(client)) if client[i].active]

        dataset_server_validation = separate_dataset(dataset, self.data_split['train'])
        data_loader_server_validation = make_data_loader({'train': dataset_server_validation}, 'client')['train']

        if cfg['diff_option'] == 'diff':
            model_0 = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model_0.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            model_0.load_state_dict(self.model_state_dict)
            model_0_dict = {k: v.cpu() for k, v in model_0.state_dict().items()}

        #######观察两两邻接矩阵开始

        # adjacency = np.zeros((10,10))

        # model_1 = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        # model_1.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))

        # model_2 = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        # model_2.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))


        # for i in range(10):
        #     for j in range(10):
        #         model_1.load_state_dict(valid_client[i].model_state_dict)
        #         model_1_dict = {k: v.cpu() for k, v in model_1.state_dict().items()}

        #         model_1_param_vector = torch.zeros([1,1])

        #         for k, v in model_1_dict.items():
        #             single_1_param_vector = torch.reshape(torch.sign(torch.sub(model_1_dict[k], model_0_dict[k])), (-1,1))
        #             model_1_param_vector = torch.cat((model_1_param_vector, single_1_param_vector), 0).int()

        #         # for k, v in model_1_dict.items():
        #         #     single_1_param_vector = torch.reshape(model_1_dict[k], (-1,1))
        #         #     model_1_param_vector = torch.cat((model_1_param_vector, single_1_param_vector), 0)

        #         model_2.load_state_dict(valid_client[j].model_state_dict)

        #         model_2_param_vector = torch.zeros([1,1])
        #         model_2_dict = {k: v.cpu() for k, v in model_2.state_dict().items()}

        #         # print('model 1 dict is', model_1_dict)
        #         # print('model 2 dict is', model_2_dict)

        #         for k, v in model_2_dict.items():
        #             single_2_param_vector = torch.reshape(torch.sign(torch.sub(model_2_dict[k], model_0_dict[k])), (-1,1))
        #             model_2_param_vector = torch.cat((model_2_param_vector, single_2_param_vector), 0).int()

        #         # for k, v in model_2_dict.items():
        #         #     single_2_param_vector = torch.reshape(model_2_dict[k], (-1,1))
        #         #     model_2_param_vector = torch.cat((model_2_param_vector, single_2_param_vector), 0)

        #         output2 = torch.count_nonzero(torch.sub(model_1_param_vector, model_2_param_vector)).item()
        #         output2 = int(output2)
        #         # print('output2 is', output2)
        #         adjacency[i,j] = output2
                
        # adjacency = adjacency.astype(int)
        # print('adjacency matix is')
        # print(adjacency)


        ###########观察邻接矩阵结束


        model_server = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model_server.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
        model_server.load_state_dict(self.model_state_dict, strict=False)
        lr = optimizer.param_groups[0]['lr']
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer_server = make_optimizer(model_server, 'local')
        optimizer_server.load_state_dict(self.optimizer_state_dict)
        model_server.train(True)

        for epoch1 in range(1, cfg['local']['num_epochs'] + 1):
            for i, input in enumerate(data_loader_server_validation):
                input = collate(input)
                # print('input22', input['data'].size(), input['data'][0])  # input22 torch.Size([10, 3, 32, 32]) 3个通道
                # break
                input_size = input['data'].size(0)
                # print('input size', input_size)   # input_size 10
                input = to_device(input, cfg['device'])
                optimizer_server.zero_grad()
                output = model_server(input)
                # print('output1', output)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model_server.parameters(), 1)
                optimizer_server.step()
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                # print('evaluation22', evaluation) #evaluation22 {'Loss': 2.296760320663452, 'Accuracy': 20.0}
                logger.append(evaluation, 'validation3', n=input_size)
        # self.optimizer_state_dict = optimizer_server.state_dict()
        # self.model_state_dict = {k: v.cpu() for k, v in model_server.state_dict().items()}
        model_server_param_vector = torch.zeros([1,1])
        model_server_dict = {k: v.cpu() for k, v in model_server.state_dict().items()}
        for k, v in model_server_dict.items():
        # for i in list(model_server.state_dict().values()):   
            if cfg['diff_option'] == 'diff':
                single_server_param_vector = torch.reshape(torch.sign(torch.sub(model_server_dict[k],model_0_dict[k])), (-1,1))
            # single_server_param_vector = torch.reshape(model_server_dict[k], (-1,1))
            elif cfg['diff_option'] == 'no-diff':
                single_server_param_vector = torch.reshape(torch.sign(model_server_dict[k]), (-1,1))
            # single_server_param_vector = torch.reshape(torch.heaviside(torch.sub(model_server_dict[k],model_0_dict[k]),torch.tensor([0])), (-1,1))
            else:
                print('diff doumeijin')
            model_server_param_vector = torch.cat((model_server_param_vector, single_server_param_vector), 0)
        # print('model server vector', model_server_param_vector, model_server_param_vector.size())
        # for i in range(model_server_param_vector.size()[0]):
        #     if model_server_param_vector[i,0] >= 0:
        #         model_server_param_vector[i,0] = 1
        #     else:
        #         model_server_param_vector[i,0] = 0
        model_server_param_vector = model_server_param_vector.int()
        # print('model server vector 0-1', model_server_param_vector, model_server_param_vector.size(),type(model_server_param_vector[0,0].item()))


        relation = []

        for m in range(len(valid_client)):
            string_client_id = str(valid_client[m].client_id.item())

            model_m = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model_m.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            model_m.load_state_dict(valid_client[m].model_state_dict, strict=False)

            model_m_param_vector = torch.zeros([1,1])
            model_m_dict = {k: v.cpu() for k, v in model_m.state_dict().items()}

            for k, v in model_m_dict.items():
                if cfg['diff_option'] == 'diff':
                    single_m_param_vector = torch.reshape(torch.sign(torch.sub(model_m_dict[k], model_0_dict[k])), (-1,1))
                # single_m_param_vector = torch.reshape(model_m_dict[k], (-1,1))
                elif cfg['diff_option'] == 'no-diff':
                    single_m_param_vector = torch.reshape(torch.sign(model_m_dict[k]), (-1,1))
                else:
                    print('diff doumeijin2')
                model_m_param_vector = torch.cat((model_m_param_vector, single_m_param_vector), 0)

            # for i in range(model_m_param_vector.size()[0]):
            #     if model_m_param_vector[i,0] >= 0:
            #         model_m_param_vector[i,0] = 1
            #     else:
            #         model_m_param_vector[i,0] = 0
            model_m_param_vector = model_m_param_vector.int()
            # print('model m vector 0-1', model_m_param_vector, model_m_param_vector.size(), type(model_m_param_vector[0,0].item()))


            output1 = torch.count_nonzero(torch.sub(model_server_param_vector, model_m_param_vector)).item()
            # output1 = torch.count_nonzero(torch.sub(model_server_param_vector, model_m_param_vector)).item()/model_server_param_vector.size()

            # print('model m vector', model_m_param_vector,  model_m_param_vector.size())
            # output1 = F.cosine_similarity(model_server_param_vector, model_m_param_vector, dim = 0)
            # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            # output = cos(model_server_param_vector, model_m_param_vector)
            # distance = torch.cdist(model_server_param_vector, model_m_param_vector, p=2)
            print('output1 is', epoch, valid_client[m].client_id, output1, type(output1), model_server_param_vector.size())
            # write_log(root, 'output1 is' + valid_client[m].client_id + output1 + type(output1) + model_server_param_vector.size())
            # print('distance is', valid_client[m].client_id, distance)
            relation.append(output1)
        print('relation is', epoch, relation)
        # write_log(root, 'relation is' + relation)
        # outlier_idx = self.find_outlier(relation, 1.5, 'high')
        outlier_idx = self.find_outlier2(relation)
        print('outlier_idx is', epoch, outlier_idx)
        # write_log(root, 'outlier_idx is' + outlier_idx)
        # max_value = max(relation)
        # max_index = relation.index(max_value)
        # print('min index is', min_index)
        # valid_client[max_index].good = False
        # print('huaidan', valid_client[max_index].client_id)
        for i in range(len(outlier_idx)):
            valid_client[outlier_idx[i]].good = False
            print('huaidan', epoch, valid_client[outlier_idx[i]].client_id)
        true_positive = 0
        false_negative = 0
        false_positive = 0
        true_negative = 0
        real_list = []
        prediction_list = []
        for i in range(len(valid_client)):
            if valid_client[i].set_malicious == True and valid_client[i].good == False:
                true_positive += 1
                real_list.append(1) # 坏人记为1,好人记为0
                prediction_list.append(1)
            elif valid_client[i].set_malicious == True and valid_client[i].good == True:
                false_negative += 1
                real_list.append(1)
                prediction_list.append(0)
            elif valid_client[i].set_malicious == False and valid_client[i].good == False:
                false_positive += 1
                real_list.append(0)
                prediction_list.append(1)
            else:
                true_negative += 1
                real_list.append(0)
                prediction_list.append(0)
        print('true positive is', epoch, true_positive, false_negative, false_positive, true_negative)
        print('real list is', epoch, real_list, prediction_list)

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
            # weight = torch.ones(len(valid_client))
            # weight = weight / weight.sum()
            ###################################
            # self.select_client_validation_set(client, dataset, optimizer, metric, logger, epoch, valid_client)
            # self.select_client_cosine(client, dataset, optimizer, metric, logger, epoch, valid_client)
            # self.select_client_zero_one(client, dataset, optimizer, metric, logger, epoch, valid_client)

            valid_good_client = [client[i] for i in range(len(client)) if client[i].active and client[i].good == True]
            for m in range(len(valid_good_client)):
                print('valid good client', epoch, valid_good_client[m].client_id)
            weight = torch.ones(len(valid_good_client))
            weight = weight / weight.sum()
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
                    for m in range(len(valid_good_client)):
                        # print('model state dict', valid_client[m].model_state_dict[k].size())
                        # if valid_client[m].good == True:
                        # print('you', valid_good_client[m].client_id)

                        # tmp_v += weight[m] * valid_good_client[m].model_state_dict[k]
                        if cfg['data_poison_method'] == 'noise-model':
                            noisy_model = add_noise_to_model(valid_client[m].model_state_dict, 0, 1)
                            tmp_v += weight[m] * noisy_model[k]

                        else:
                            tmp_v += weight[m] * valid_good_client[m].model_state_dict[k]
                    v.grad = (v.data - tmp_v).detach()
            global_optimizer.step()
            self.global_optimizer_state_dict = global_optimizer.state_dict()
            self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        for i in range(len(client)):
            client[i].active = False
            client[i].good = True
            if cfg['adversarial_ratio'].split('-')[0] == 'channel':
                client[i].set_malicious = False
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
        self.set_malicious = False      #是否需要污染数据，使其成为坏人
        self.good = True    #经过筛选判定为好人

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
