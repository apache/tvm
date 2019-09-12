import os
import time
import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

########new 
from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel
from tvm.autotvm.tuner.model_based_tuner import ModelBasedTuner
from tvm.autotvm.measure import MeasureInput, create_measure_batch
import argparse
#################################################################
# Define Network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`nnvm.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--trail', type=int, default=100)
parser.add_argument('--feature-type', type=str, choices=['itervar', 'curve'], default='itervar')
parser.add_argument('--layer', type=int, default=0)
args = parser.parse_args()


def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        net, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        net, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        net, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        net, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        net, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        net, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
    else:
        raise ValueError("Unsupported network: " + name)

    return net, params, input_shape, output_shape

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we apply some configurations.

#### DEVICE CONFIG ####
target = tvm.target.cuda()

#### TUNING OPTION ####
network = 'resnet-18'
log_file = "%s.log" % network
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        #runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        runner=autotvm.LocalRunner(number=10, repeat=1, timeout=20, min_repeat_ms=2000),
        #runner=autotvm.RPCRunner(
        #    '1080ti',  # change the device key to your key
        #    '0.0.0.0', 9190,
        #    number=20, repeat=3, timeout=4, min_repeat_ms=150)
    ),
}



################## I do not think we need any tuning for now, let's do random search first.
def feature_extraction(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    net, params, input_shape, out_shape = get_network(network, batch_size=args.batch_size)
    tasks = autotvm.task.extract_from_program(net, target=target,
                                            params=params, ops=(relay.op.nn.conv2d,))

    print("Create XGBoost Cost Model so that we can extract feature and its corresponding description")
    current_testing_task = args.layer
    
    ######### without gradient?????????
#     if try_winograd==True:
    if True:
        try:  # try winograd template
            tsk = autotvm.task.create(tasks[current_testing_task].name, tasks[current_testing_task].args,
                                      tasks[current_testing_task].target, tasks[current_testing_task].target_host, 'winograd')
            input_channel = tsk.workload[1][1]
            if input_channel >= 64:
                tasks[current_testing_task] = tsk
        except Exception:
            pass
    task = tasks[current_testing_task]
#     print('Our current testing task is: ', task)
    
    cost_model = XGBoostCostModel(task,
                                  feature_type=args.feature_type,
                                  loss_type='rank',
                                  num_threads=None,
                                  log_interval=50 // 2)
    # print(cost_model.space)
    total_space = 844800
    # changing batch size won't change the number of configs.
    # for tasks[0], there are 844800 configs, which is a large number to explore.
    # For Now, let's extract 2000 configs from the entire space.
    testing_number = args.trail
    
    ######## indexes are the same for different batch size
    indexes_file_name = 'data/indexes_saving_layer_'+str(args.layer)+'.npy'
    if not os.path.exists(indexes_file_name):
        indexes = np.random.choice(range(total_space), testing_number, replace=False)
        np.save(indexes_file_name, indexes)
    else:
        indexes = np.load(indexes_file_name)
    
    ####### features 
    features_file_name = 'data/features_saving_'+args.feature_type+'_layer_'+str(args.layer)+'_batchsize_'+str(args.batch_size)+'.npy'
    if not os.path.exists(features_file_name):
        features_testing = cost_model._get_feature(indexes)
        np.save(features_file_name, features_testing)
    
    

    #################################
    # Until now, we can extract feature and the correspong index,
    # our next step is to measure the time
    #################################
    print("Let's measure the time Now!!!!!!!!!!!!")
    measure_batch = create_measure_batch(task, tuning_opt["measure_option"])
    n_parallel = getattr(measure_batch, 'n_parallel', 1)
    early_stopping = tuning_opt["early_stopping"]
    latency_output = np.zeros(testing_number)
    flops_output = np.zeros(testing_number)
    
    Current_list = list(np.arange(testing_number))
    ###### Actually Failure Conf cannot run. I do not know how to deal with that!
#     while Current_list:
    Success_list = []
    Failure_list = []
    for i in Current_list:
        configs = [task.config_space.get(indexes[i])]
        inputs = [MeasureInput(task.target, task, config) for config in configs]
        results = measure_batch(inputs)
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
#                 print(res.costs)
                print('Success of %d-th configuration ' % i)
                Success_list.append(i)
                latency_output[i] = np.mean(res.costs)
                flops_output[i] = inp.task.flop / np.mean(res.costs)
            else:
                print('Failure of %d-th configuration ' % i)
                Failure_list.append(i)
#             time.sleep(0.5)
#     Current_list = Tmp_list
    print('\n***********************\n Left Item: %d \n********************' % len(Failure_list))
    
    # latency is the same for different feature
    latency_file_name = 'data/latency_saving'+'_layer_'+str(args.layer)+'_batchsize_'+str(args.batch_size)+'.npy'
    np.save(latency_file_name, latency_output)
    
    # latency is the same for different feature
    flops_file_name = 'data/flops_saving'+'_layer_'+str(args.layer)+'_batchsize_'+str(args.batch_size)+'.npy'
    np.save(flops_file_name, flops_output)
    
    # success is the same for different feature
    success_file_name = 'data/success_saving'+'_layer_'+str(args.layer)+'_batchsize_'+str(args.batch_size)+'.npy'
    np.save(success_file_name, Success_list)

feature_extraction(tuning_option)
