task_name='hybrid_superresolution_pipeline'

## multiple gpus
n_gpu=1
batchsize=30
## number of cpus for loading data
n_cpu=14

## learning rate and batch size
learning_rate=1e-4

bin_len=32e-12 * 3e8

## parameters about the resolution of different signals and images
resolution_ver=32
resolution_hor=32

resolution_down_ver=8
resolution_down_hor=8

sig_super_scale=4

##test
is_testing = True
dataset_name='Experiment18m'
Output_path = './OUTPUTS_%s/' % (dataset_name)
sample_size_test = 5
sample_id = 0
wall_size = 0.82
sampling_coeff = 3.5
cycles = 7.0
time_span = 512
net1_weight_path = './weight/net1_18m.pkl'
net2_weight_path = './weight/net2_18m.pkl'

