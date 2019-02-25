import numpy as np
from argparse import ArgumentParser
import os
import json
from string import digits

try:
    from tempfile import TemporaryDirectory
except ImportError:
    import tempfile
    import shutil

    class TemporaryDirectory(object):
        """Context manager for tempfile.mkdtemp()"""

        def __enter__(self):
            self.name = tempfile.mkdtemp()
            return self.name

        def __exit__(self, exc_type, exc_value, traceback):
            shutil.rmtree(self.name)

def pth_to_json(model_path, input_shapes, outdir):
    r'''
    Load a model in PyTorch and save it as .json (symbol graph) and .npz (weights).
    '''
    import torch
    from torch.onnx.utils import _optimize_graph, _model_to_graph
    from torch.onnx import OperatorExportTypes
    pytorch_version = '.'.join(torch.__version__.split('.')[:2])
    def fancy_split(s, delim):
        tokens = []
        start = 0
        for end in range(len(s) + 1):
            if end == len(s) or start != end and s[end] == delim and s[end-1] not in digits or end == len(s):
                tokens.append(s[start:end].split(':')[1].strip())
                start = end + 1
        return tokens
    dtype_map = {
        torch.float64: 'Double',
        torch.float32: 'Float',
        torch.int64: 'Long',
        torch.int32: 'Int',
        np.dtype(np.float64): 'Double',
        np.dtype(np.float32): 'Float',
        np.dtype(np.int64): 'Long',
        np.dtype(np.int32): 'Int',
    }
    input_names = sorted(input_shapes)
    model_inputs = tuple(torch.zeros(input_shapes[k]).float() for k in input_names)
    num_inputs = len(model_inputs)
    if pytorch_version == '0.4':
        model = torch.load(model_path).float()
        g, _, _ = _model_to_graph(model, model_inputs, None, OperatorExportTypes.ONNX)
    elif pytorch_version == '1.0':
        model = torch.jit.load(model_path).float().eval()
        g = _optimize_graph(model.graph, OperatorExportTypes.ONNX)
    params = {}
    for (i, (k, v)) in enumerate(model.state_dict().items()):
        params[str(i + num_inputs)] = v.numpy()
    for i, shape in enumerate(input_shapes):
        params[input_names[i]] = model_inputs[i].numpy()
    sym = {}
    names = {}
    for index, tensor_node in enumerate(g.inputs()):
        name = str(index)
        names[tensor_node.uniqueName()] = name
        if index < num_inputs:
            a = model_inputs[index].numpy()
            name = input_names[index]
        else:
            a = params[name]
        shape = a.shape
        dtype = dtype_map[a.dtype]
        sym[name] = {
            'shape': shape,
            'dtype': dtype,
            'is_input': index < num_inputs,
        }
    for index, op_node in enumerate(g.nodes(), index + 1):
        name = str(index)
        names[next(op_node.outputs()).uniqueName()] = name
        if index < num_inputs:
            name = input_names[index]
        op_name = op_node.kind().split('::')[-1]
        attrs = {k: op_node[k] for k in op_node.attributeNames()}
        sym[name] = {}
        if 'value' in attrs and not isinstance(attrs['value'], (float, int)):
            value = attrs['value'].numpy()
            attrs['value'] = value.tolist()
            sym[name]['dtype'] = dtype_map[value.dtype]
            sym[name]['shape'] = value.shape
        inputs = [names[inp.uniqueName()] for inp in op_node.inputs()]
        for i in range(len(inputs)):
            if eval(inputs[i]) < num_inputs:
                inputs[i] = input_names[eval(inputs[i])]
        tokens = fancy_split(str(op_node).split('=')[0], ',')
        if eval(name) < num_inputs:
            name = input_names[eval(name)]
        rtype = tokens[0]
        sym[name].update({
            'op_name': op_name,
            'attrs': attrs,
            'inputs': inputs,
            'rtype': rtype,
        })
        if '(' in rtype:
            dtype = rtype[:rtype.index('(')]
            shape = eval(rtype[rtype.index('('):])
            sym[name]['dtype'] = dtype
            sym[name]['shape'] = shape
    with open(os.path.join(outdir, 'model.json'), 'w') as f:
        json.dump(sym, f)
    np.savez(os.path.join(outdir, 'model.npz'), **params)

def json_to_nnvm(model_path):
    r'''
    Load a PyTorch model as .json and .npz and return as NNVM symbol graph
    '''
    import tvm
    from nnvm.frontend.onnx import _get_convert_map
    from .. import symbol
    inv_dtype_map = {
        'Double': np.dtype(np.float64),
        'Float': np.dtype(np.float32),
        'Long': np.dtype(np.int64),
        'Int': np.dtype(np.int64),
    }
    with open(os.path.join(model_path, 'model.json'), 'r') as f:
        sym_raw = json.load(f)
    params = dict(np.load(os.path.join(model_path, 'model.npz')))
    sym = {}
    inputs = set()
    for k, v in params.items():
        dtype = inv_dtype_map[sym_raw[k]['dtype']]
        params[k] = tvm.nd.array(v.astype(dtype))
        sym[k] = symbol.Variable(name=k, shape=v.shape)
        if sym_raw[k]['is_input']:
            inputs.add(k)
            # TODO: support -1 reshape
            batch_size = sym_raw[k]['shape'][0]
    for k in inputs:
        params.pop(k)
    var_op_dtypes = {}
    for k in sorted(inputs) + sorted([k for k in sym_raw if 'is_input' not in sym_raw[k] or not sym_raw[k]['is_input']], key=lambda k: int(k)):
        v = sym_raw[k]
        if 'op_name' not in v:
            continue
        op_name = v['op_name']
        attrs = v['attrs']
        if op_name == 'max_pool2d_with_indices':
            op_name = 'MaxPool'
            attrs.pop('dilation')
            inputs = [sym[l] for l in sym_raw[k]['inputs']]
            op = _get_convert_map(1)[op_name](inputs, attrs, params)
            sym[k] = last_op = op[0]
        elif op_name == 'Constant':
            dtype = inv_dtype_map[v['dtype']]
            init = np.array(v['attrs']['value'], dtype=dtype)
            shape = (1,) + tuple(v['shape'])
            sym[k] = last_op = symbol.Variable(name=k,
                                               init=init,
                                               shape=shape)
            var_op_dtypes[k] = v['dtype']
            last_op = sym[k]
        elif op_name == 'Shape':
            init = np.array(sym_raw[v['inputs'][0]]['shape'])
            shape = init.shape
            sym[k] = last_op = symbol.Variable(name=k,
                                               init=init,
                                               shape=shape)
            var_op_dtypes[k] = 'int64'
        elif op_name.lower() == 'reshape':
            op_name = op_name.capitalize()
            # TODO: implement reshape with two symbolic inputs
            shape = np.array(v['shape'])
            num_features = int(abs(shape.prod()) / batch_size)
            attrs['shape'] = [batch_size, num_features]
            inputs = [sym[l] for l in sym_raw[k]['inputs']]
            op = _get_convert_map(1)[op_name](inputs, attrs, params)
            sym[k] = last_op = op[0]
        else:
            if op_name in {'Add', 'Sub', 'Mul', 'Unsqueeze'}:
                attrs['pytorch'] = True
            inputs = [sym[l] for l in sym_raw[k]['inputs']]
            op = _get_convert_map(1)[op_name](inputs, attrs, params)
            sym[k] = last_op = op[0]
    input_name = last_op.list_input_names()[0]
    dtype_dict = {input_name: 'float32'}
    for k, v in params.items():
        dtype_dict[k] = v.dtype
    dtype_dict.update(var_op_dtypes)
    return last_op, params, dtype_dict

def from_pytorch(model_path, input_shapes):
    import tvm
    from nnvm.frontend.onnx import _get_convert_map
    from .. import symbol
    import torch
    from torch.onnx.utils import _model_to_graph
    from torch.onnx import OperatorExportTypes
    with TemporaryDirectory() as tmp:
        pth_to_json(model_path, input_shapes, tmp)
        return json_to_nnvm(tmp)
