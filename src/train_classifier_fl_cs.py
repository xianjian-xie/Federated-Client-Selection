import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg, process_args
from data import fetch_dataset, shuffle_dataset_target, split_dataset, make_data_loader, separate_dataset, make_batchnorm_dataset, \
    make_batchnorm_stats, shuffle_dataset_target
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    celist = [0,1,2,3,4]
    print('ceshi', celist[2:2])
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, 'global')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
    optimizer = make_optimizer(model, 'local')
    scheduler = make_scheduler(optimizer, 'global')
    batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
    data_split = split_dataset(dataset, cfg['num_clients'] + 1, cfg['data_split_mode'])
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy'], 'validation': ['Loss', 'Accuracy']} )
    result = resume(cfg['model_tag'], resume_mode=cfg['resume_mode'])
    # print('dataset',type(dataset['train']),len(dataset['train'].target),dataset['train'].target)
    if result is None:
        last_epoch = 1
        server = make_server(model, data_split)
        client = make_client(model, data_split)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = result['epoch']
        data_split = result['data_split']
        server = result['server']
        client = result['client']
        optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        logger = result['logger']
    # 
    # print('datasplit',data_split['train'][cfg['num_clients']],len(data_split['train'][cfg['num_clients']]),cfg['num_clients'])
    # server_split = {'train': data_split['train'][cfg['num_clients']], 'test': data_split['test'][cfg['num_clients']]}
    # dataset_server_validation = separate_dataset(dataset['train'], server_split['train'])
    # data_loader_server_validation = make_data_loader({'train': dataset_server_validation}, 'client')['train']
    # print('length', len(data_loader_server_validation))
    # length 50, 遍历loader一共49个
    # 
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        train_client(dataset['train'], server, client, optimizer, metric, logger, epoch)
        # server.select_client(client, dataset['train'], optimizer, metric, logger, epoch)
        server.update(client, dataset['train'], optimizer, metric, logger, epoch )
        scheduler.step()
        model.load_state_dict(server.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test(data_loader['test'], test_model, metric, logger, epoch)
        result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(), 'data_split': data_split, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def make_server(model, data_split):
    server = Server(model, {'train': data_split['train'][cfg['num_clients']], 'test': data_split['test'][cfg['num_clients']]})
    return server


def make_client(model, data_split):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model, {'train': data_split['train'][m], 'test': data_split['test'][m]})
    return client


def train_client(dataset, server, client, optimizer, metric, logger, epoch):
    logger.safe(True)
    num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))  # 100 clients只激活了0.1，即10个
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    print('client id1', client_id)
    for i in range(num_active_clients):
        client[client_id[i]].active = True
    server.distribute(client)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    # print('active_clients', num_active_clients) #10
    for i in range(num_active_clients):
        m = client_id[i]
        print('m is', m) # m 是 第i个client的id号
        dataset_m = separate_dataset(dataset, client[m].data_split['train']) #separate 分 data, target
        # print('client_m_data_split_train_len', len(client[m].data_split['train']))
        # print('client_m_data_split_train', client[m].data_split['train']) # 第m个client包含的数据的idx号
        # 按照 idx, 把数据集分割成id, data, target
        # print('dataset1_len',len(dataset_m.target))
        # print('dataset1',dataset_m.target)
        ############################################################
        if i==1:
            dataset_m = shuffle_dataset_target(dataset_m)
        ############################################################
        # print('dataset2',dataset_m.target)
        if dataset_m is not None:
            client[m].active = True
            client[m].train(dataset_m, lr, metric, logger)
        else:
            client[m].active = False
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                             'Learning rate: {:.6f}'.format(lr),
                             'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            # print('metric name', metric.metric_name['train']) # ['Loss', 'Accuracy']
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
