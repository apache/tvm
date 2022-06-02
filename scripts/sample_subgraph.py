'''
Sample subgraph from Relay IR models and randomly sample from the mutations
of subgraphs.
'''

import os
import pickle

from tvm import meta_schedule as ms

def extract_tasks(name, input_shape, target='llvm'):
    '''
    Task extraction on the Relay IR model to get all subgraphs.
    '''
    dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
    relay_ir_dir = os.path.join(dataset_dir, 'relay_ir')
    params_dir = os.path.join(dataset_dir, 'params')
    mod_filename = name + '_' + '_'.join(str(i) for i in input_shape) + '_relay_ir.pickle'
    with open(os.path.join(relay_ir_dir, mod_filename), 'rb') as file:
        mod = pickle.load(file)
    params_filename = name + '_' + '_'.join(str(i) for i in input_shape) + '_params.pickle'
    with open(os.path.join(params_dir, params_filename), 'rb') as file:
        params = pickle.load(file)
    extracted_tasks = ms.extract_task_from_relay(mod, target=target, params=params)

    return extracted_tasks


if __name__ == '__main__':
    all_subgraphs = extract_tasks('resnet_18', [1, 3, 224, 224])
