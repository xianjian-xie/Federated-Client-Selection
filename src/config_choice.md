---
# control
control:
  data_name: CIFAR10
  model_name: wresnet28x2, cnn
  num_clients: '100'
  active_rate: '0.1'
  data_split_mode: 'iid'
  data_poison_method: 'target', 'dataset', 'model', 'target-dataset', 'none'
  adversarial_ratio: 'local-0.1', 'channel-0.1', 'none'
  # local指100个里面有10个坏蛋，channel指每次active的10个必有1个坏蛋 
  selection_method: 'valid-acc-cluster', 'valid-cos-cluster', 'valid-mask-cluster', 'true', 'none'
  diff_option: 'no-diff', 'diff'
  # detection_method: 'cluster' //加一个outlier detection method, not quantile， 查adversarial detection with clustering
  cluster用高斯mixture model, 伯努利mixture model, dili process, 
# experiment
pin_memory: True
num_workers: 0
init_seed: 0
num_experiments: 1
log_interval: 0.25
device: cuda
world_size: 1
resume_mode: 1
verbose: True
