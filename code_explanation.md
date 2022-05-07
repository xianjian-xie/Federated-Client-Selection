config.yml
# control
control:
  data_name: CIFAR10
  model_name: wresnet28x2
  num_clients: '100'
  active_rate: '0.1'
  data_split_mode: 'iid'
# experiment
pin_memory: True    ?
num_workers: 0  ?
init_seed: 0
num_experiments: 1
log_interval: 0.25  ?
device: cuda
world_size: 1   ?
resume_mode: 0  ?
verbose: False  ?


https://blog.csdn.net/qinyilang/article/details/5484415


1. **Command:**
   
   1. 
   conda activate recsys_ae_conda python=3.8
   <!-- train_classifier_fl -->
   !python train_classifier_fl.py --control_name CIFAR10_wresnet28x2_100_0.03_iid --pin_memory True --num_workers 0 --init_seed 0 --num_experiments 1 --log_interval 0.25 --device cpu --resume_mode 0 --verbose False
  <!-- train_classifier_fl 标准-->
   python train_classifier_fl_cs.py --control_name CIFAR10_wresnet28x2_100_0.1_iid --pin_memory True --num_workers 0 --init_seed 0 --num_experiments 1 --log_interval 0.25 --device cpu --resume_mode 0 --verbose False

   python train_classifier_fl_cs.py --control_name CIFAR10_wresnet28x2_100_0.03_iid --pin_memory True --num_workers 0 --init_seed 0 --num_experiments 1 --log_interval 0.25 --device cpu --resume_mode 0 --verbose False


   


1）iteration：表示1次迭代（也叫training step），每次迭代更新1次网络结构的参数；（2）batch-size：1次迭代所使用的样本量；（3）epoch：1个epoch表示过了1遍训练集中的所有样本。值得注意的是，在深度学习领域中，常用带mini-batch的随机梯度下降算法（Stochastic Gradient Descent, SGD）训练深层结构，它有一个好处就是并不需要遍历全部的样本，当数据量非常大时十分有效。此时，可根据实际问题来定义epoch，例如定义10000次迭代为1个epoch，若每次迭代的batch-size设为256，那么1个epoch相当于过了2560000个训练样本。

