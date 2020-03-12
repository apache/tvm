# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines, redefined-builtin
"""Relay to ONNX serialization """

import numpy
import onnx
import onnx.utils
from onnx import numpy_helper, OperatorSetIdProto, defs
import tvm
from tvm.autotvm.graph_tuner.utils.traverse_graph import _expr2graph_impl
from tvm.relay.expr import Call, TupleGetItem, Var, Constant, Tuple

ONNX_OPSET_VERSONS_SUPPORTED = [11]


def tvm_array_to_list(arr):
    return tuple(x.value for x in arr)


def get_onnx_version():
    return onnx.__version__


def add_input(data, name, model_container):
    dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[data.dtype]
    tensor_value_info = onnx.helper.make_tensor_value_info(name, dtype, shape=data.shape)
    model_container.add_inputs([tensor_value_info])
    data_tensor = numpy_helper.from_array(data, name)
    model_container.add_initializers([data_tensor])


class OpConverter(object):
    """ Operator converter Base Class.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        """convert Relay attributes to ONNX attributes.
           The derived classes should implement this method
           if attributes are required by the operator
           otherwise by default no attributes are passed
        """
        return {}

    @classmethod
    def convert(cls, node, model_container, node_list):
        attrs = cls.convert_attributes(node['node'].attrs)
        node = onnx.helper.make_node(cls.__name__,
                                     node['input_names'],
                                     node['output_names'],
                                     **attrs)
        model_container.add_nodes([node])


def rename(op_name):
    """ This method creates dynamic operator of name op_name with empty attributes
    """
    return type(op_name, (OpConverter,), {})


class Reshape(object):
    """ Operator converter for Reshape.
    """

    @classmethod
    def convert(cls, node, model_container, node_list):
        """Converts Relay operator Reshape to ONNX operator.
           Relay operator accepts shape as attribute but ONNX operator
           accepts it as a input.
        """

        shape = numpy.asarray([a.value for a in node['node'].attrs.newshape],
                              dtype=numpy.int64)
        input_name = 'shape{}'.format(node['output_names'][0])
        node = onnx.helper.make_node(cls.__name__, [node['input_names'][0], input_name],
                                     node['output_names'])
        model_container.add_nodes([node])
        add_input(shape, input_name, model_container)


class Conv(OpConverter):
    """ Operator converter for Conv.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'group': attrs.get_int("groups"),
            'pads': attrs.get_int_tuple("padding"),
            'strides': attrs.get_int_tuple("strides"),
            'dilations': attrs.get_int_tuple("dilation"),
            'kernel_shape': attrs.get_int_tuple("kernel_size"),
        }


class MaxPool(OpConverter):
    """ Operator converter for MaxPool.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'pads': attrs.get_int_tuple("padding") + attrs.get_int_tuple("padding"),
            'strides': attrs.get_int_tuple("strides"),
            'kernel_shape': attrs.get_int_tuple("pool_size"),
        }


class Transpose(OpConverter):
    """ Operator converter for Transpose.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {'perm': attrs.get_int_tuple("axes")} if attrs["axes"] else {}


class MatMul(OpConverter):
    """ Operator converter for MatMul.
    """

    @classmethod
    def convert(cls, node, model_container, node_list):
        output_name = 'inter{}'.format(node['output_names'][0])
        transpose_node = onnx.helper.make_node(Transpose.__name__,
                                               [node['input_names'][1]],
                                               [output_name],
                                               **{'perm': (1, 0)})
        model_container.add_nodes([transpose_node])

        inputs = [node['input_names'][0], output_name]
        matmul_node = onnx.helper.make_node(cls.__name__, inputs, node['output_names'])
        model_container.add_nodes([matmul_node])


class Flatten(OpConverter):
    """ Operator converter for Flatten.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'axis': 1,
        }


class BatchNormalization(OpConverter):
    """ Operator converter for BatchNormalization.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'epsilon': float(attrs.get_str('epsilon')),
            'axis': float(attrs.get_int('axis')),
        }

    @classmethod
    def convert(cls, node, model_container, node_list):
        """Converts Relay operator batch_norm to ONNX operator.
           Relay operator has property axis to handle data in NHWC format.
        """
        attrs = cls.convert_attributes(node['node'].attrs)
        transpose_out_name = node['input_names'][0]
        output_names = node['output_names']

        # axis==3 means channel is specified along the 3rd axis
        if attrs['axis'] == 3:
            transpose_out_name = 'transpose_{}'.format(node['output_names'][0])
            node_transposed = onnx.helper.make_node(Transpose.__name__,
                                                    [node['input_names'][0]],
                                                    [transpose_out_name],
                                                    **{'perm': [0, 3, 1, 2]})
            model_container.add_nodes([node_transposed])
            output_names = ['batch_norm_{}'.format(node['output_names'][0])]

        batch_norm_node = onnx.helper.make_node(cls.__name__,
                                                [transpose_out_name] + node['input_names'][1:],
                                                output_names,
                                                **{'epsilon': attrs['epsilon']})
        model_container.add_nodes([batch_norm_node])

        if attrs['axis'] == 3:
            node_transposed = onnx.helper.make_node(Transpose.__name__,
                                                    output_names,
                                                    node['output_names'],
                                                    **{'perm': [0, 2, 3, 1]})
            model_container.add_nodes([node_transposed])


class Dropout(OpConverter):
    """ Operator converter for Dropout.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'ratio': float(attrs.get_str('rate')),
        }


class AveragePool(MaxPool):
    """ Operator converter for AveragePool.
    """


class Concat(OpConverter):
    """ Operator converter for Concat.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'axis': attrs.get_int("axis"),
        }


class BiasAdd(OpConverter):
    """ Operator converter for BiasAdd.
    """

    @classmethod
    def convert(cls, node, model_container, node_list):
        input_node = node_list[node['inputs'][0][0]]
        data_ndim = len(input_node['types'][0].shape)
        axis = node['node'].attrs.get_int("axis")
        if axis < 0:
            axis = axis + data_ndim
        new_axes = data_ndim - axis - 1
        if new_axes:
            output_name = 'inter{}'.format(node['output_names'][0])
            unsqueeze_node = onnx.helper.make_node('Unsqueeze',
                                                   [node['input_names'][1]],
                                                   [output_name],
                                                   **{'axes': tuple(range(1, new_axes + 1))})
            model_container.add_nodes([unsqueeze_node])
        else:
            output_name = node['input_names'][1]

        inputs = [node['input_names'][0], output_name]
        matmul_node = onnx.helper.make_node('Add', inputs, node['output_names'])
        model_container.add_nodes([matmul_node])


class ReduceMean(OpConverter):
    """ Operator converter for ReduceMean.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'axes': attrs.axis,
            'keepdims': 0 if bool(attrs.get_int("keepdims", 0)) is False else 1
        }

    @classmethod
    def convert(cls, node, model_container, node_list):
        input_node = node_list[node['inputs'][0][0]]
        shape = input_node['types'][0].shape
        axis = node['node'].attrs.axis
        axis = list(range(shape.size())) if not axis else tvm_array_to_list(axis)
        exclude = 0 if not bool(node['node'].attrs.exclude) else 1
        keepdims = 0 if not bool(node['node'].attrs.keepdims) else 1
        if exclude:
            all_axis = list(range(len(shape)))
            axis = set(all_axis) - set(axis)

        node = onnx.helper.make_node(cls.__name__,
                                     node['input_names'],
                                     node['output_names'],
                                     **{"axes": axis,
                                        "keepdims": keepdims})
        model_container.add_nodes([node])


class Pad(OpConverter):
    """ Operator converter for Pad.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        before = []
        after = []
        for axis_pads in attrs.pad_width:
            before.append(axis_pads[0])
            after.append(axis_pads[1])
        pads = before + after
        pads = numpy.asarray(pads, dtype=pads[0].dtype)
        return {
            'pads': pads,
            'mode': attrs.get_str('pad_mode'),
            'constant_value': attrs.pad_value
        }

    @classmethod
    def convert(cls, node, model_container, node_list):
        """Converts Relay operator Pad to ONNX operator.
           Relay operator accepts pads as attribute but ONNX operator
           accepts it as a input.
        """
        attrs = cls.convert_attributes(node['node'].attrs)

        data = numpy.asarray(attrs['pads'], dtype=attrs['pads'][0].dtype).astype(numpy.int64)
        input_name = 'pads_{}'.format(node['output_names'][0])
        value = numpy.dtype(node['types'][0].dtype).type(attrs['constant_value'])
        input_value_name = 'value_{}'.format(node['output_names'][0])
        add_input(data, input_name, model_container)
        add_input(value, input_value_name, model_container)

        input_names = [node['input_names'][0], input_name, input_value_name]
        node = onnx.helper.make_node(cls.__name__, input_names, node['output_names'])
        model_container.add_nodes([node])


class Softmax(OpConverter):
    """ Operator converter for SoftMax.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'axis': attrs.axis,
        }


class Squeeze(OpConverter):
    """ Operator converter for Squeeze.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'axes': attrs.axis,
        }

    @classmethod
    def convert(cls, node, model_container, node_list):
        input_node = node_list[node['inputs'][0][0]]
        shape = input_node['types'][0].shape
        axis = node['node'].attrs.get_int("axis")
        if not axis:
            axis = []
            for axis_idx, val in enumerate(shape):
                if val.value == 1:
                    axis.append(axis_idx)
        else:
            axis = node['node'].attrs.get_int_tuple("axis")

        node = onnx.helper.make_node(cls.__name__,
                                     node['input_names'],
                                     node['output_names'],
                                     **{"axes": axis})
        model_container.add_nodes([node])


class Slice(OpConverter):
    """ Operator converter for Slice.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'starts': attrs.get_int_tuple('begin'),
            'ends': attrs.get_int_tuple('end'),
            'steps': attrs.get_int_tuple('strides')
        }

    @classmethod
    def convert(cls, node, model_container, node_list):
        attrs = cls.convert_attributes(node['node'].attrs)

        input_node = node_list[node['inputs'][0][0]]
        shape = input_node['types'][0].shape
        starts = list(attrs['starts'])
        ends = list(attrs['ends'])
        for i in range(len(starts), len(shape)):
            starts.append(0)
        for i in range(len(ends), len(shape)):
            ends.append(shape[i] + 1)

        starts = numpy.asarray(starts).astype(numpy.int64)
        starts_name = 'starts_{}'.format(node['output_names'][0])
        add_input(starts, starts_name, model_container)

        ends = numpy.asarray(ends).astype(numpy.int64)
        ends_name = 'ends_{}'.format(node['output_names'][0])
        add_input(ends, ends_name, model_container)

        input_names = node['input_names'] + [starts_name, ends_name]

        if attrs['steps']:
            axes = list(range(len(shape)))
            attrs['axes'] = axes
            assert len(axes) == len(attrs['steps']), "axes and steps should be of same size"

            steps = numpy.asarray(attrs['steps']).astype(numpy.int64)
            steps_name = 'steps_{}'.format(node['output_names'][0])
            add_input(steps, steps_name, model_container)

            axes = numpy.asarray(attrs['axes']).astype(numpy.int64)
            axes_name = 'axes_{}'.format(node['output_names'][0])
            add_input(axes, axes_name, model_container)

            input_names = input_names + [axes_name, steps_name]

        slice_node = onnx.helper.make_node(cls.__name__,
                                           input_names,
                                           node['output_names'])
        model_container.add_nodes([slice_node])


class ConstantOfShapeZeros(OpConverter):
    """ Operator converter for ConstantOfShape.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'value': 0
        }

    @classmethod
    def convert(cls, node, model_container, node_list):
        attrs = cls.convert_attributes(node['node'].attrs)
        input_node = node_list[node['inputs'][0][0]]
        shape = input_node['types'][0].shape
        dtype = input_node['types'][0].dtype
        input_shape_name = 'shape_{}'.format(node['output_names'][0])
        shape = numpy.asarray(shape).astype(numpy.int64)
        add_input(shape, input_shape_name, model_container)

        dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(dtype)]
        tensor_value = onnx.helper.make_tensor("value", dtype,
                                               [1], [attrs['value']])

        node = onnx.helper.make_node('ConstantOfShape',
                                     [input_shape_name],
                                     node['output_names'],
                                     **{'value': tensor_value})
        model_container.add_nodes([node])


class ConstantOfShapeOnes(ConstantOfShapeZeros):
    """ Operator converter for ConstantOfShape.
    """

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'value': 1
        }


relay_to_onnx_op_mapping = {
    'reshape': Reshape,
    'nn.conv2d': Conv,
    'add': rename('Add'),
    'nn.relu': rename('Relu'),
    'transpose': Transpose,
    'nn.dense': MatMul,
    'nn.max_pool2d': MaxPool,
    'nn.batch_flatten': Flatten,
    'multiply': rename('Mul'),
    'nn.bias_add': BiasAdd,
    'nn.batch_norm': BatchNormalization,
    'nn.global_avg_pool2d': rename('GlobalAveragePool'),
    'concatenate': Concat,
    'nn.dropout': Dropout,
    'nn.avg_pool2d': AveragePool,
    'divide': rename('Div'),
    'mean': ReduceMean,
    'nn.pad': Pad,
    'nn.softmax': Softmax,
    'squeeze': Squeeze,
    'strided_slice': Slice,
    'greater': rename('Greater'),
    'less': rename('Less'),
    'equal': rename('Equal'),
    'zeros_like': ConstantOfShapeZeros,
    'ones_like': ConstantOfShapeOnes,
    'subtract': rename('Sub')
}


class ModelContainer(object):
    """ A container class to hold  different attributes of ONNX model graph
    """

    def __init__(self, name, opset_version):
        self._name = name
        self._opset_version = opset_version
        self._inputs = []
        self._outputs = []
        self._nodes = []
        self._initializers = []

    def add_inputs(self, inputs):
        self._inputs.extend(inputs)

    def add_outputs(self, outputs):
        self._outputs.extend(outputs)

    def add_nodes(self, nodes):
        self._nodes.extend(nodes)

    def add_initializers(self, initializers):
        self._initializers.extend(initializers)

    def _get_opsets(self):
        opsets = []
        imp = OperatorSetIdProto()
        imp.version = self._opset_version
        opsets.append(imp)
        return opsets

    def make_model(self):
        """ Creates the onnx model from the graph """
        onnx_graph = onnx.helper.make_graph(
            self._nodes,
            self._name,
            self._inputs,
            self._outputs,
            self._initializers
        )
        kwargs = {}
        kwargs["opset_imports"] = self._get_opsets()
        kwargs["producer_name"] = 'TVM Relay'
        kwargs["producer_name"] = tvm.__version__

        return onnx.helper.make_model(onnx_graph, **kwargs)


class RelayToONNXConverter(object):
    """A helper class converting topologically sorted Relay nodes to ONNX model

    Parameters
    ----------
    name : str
       name of the model

    node_list : list
        topologically sorted Relay Node entry list
    """

    def __init__(self, name, node_list, params, opset_version):
        self._name = {}
        self._mc = ModelContainer(name, opset_version)
        self._node_list = node_list
        self._params = params

    def convert_to_onnx(self):
        """ Loop through topologically sorted list of Relay nodes and generate a ONNX model"""
        for idx, node_entry in enumerate(self._node_list):
            out_idx = idx
            node = node_entry['node']
            if isinstance(node, Call):
                self._add_node(node_entry, idx)
            elif isinstance(node, Var):
                self._add_input(node_entry, idx)
            elif isinstance(node, Constant):
                self._add_constant_input(node_entry, idx)
            elif isinstance(node, (TupleGetItem, Tuple)):
                out_idx = idx - 1  # TODO: Need to work on this.
                # No equivalent ONNX operator found yet
            else:
                raise NotImplementedError("Relay Node of type {0} is not "
                                          "implemented yet".format(type(node)))

            if idx == len(self._node_list) - 1:
                self._add_output(self._node_list[out_idx], out_idx)

        model = self._mc.make_model()
        polished_model = onnx.utils.polish_model(model)
        return polished_model

    def _tuple_to_name(self, input):
        """convert tuple of node indexes to string"""
        return 'node_{0}'.format(input[0])

    def _add_node(self, node_entry, idx):
        """Convert Relay operator node to ONNX operator and add it to container nodes list"""
        if node_entry['op'].name not in relay_to_onnx_op_mapping:
            raise NotImplementedError("Currently the operator '{0}' is "
                                      "not supported.".format(node_entry['op'].name))

        converter = relay_to_onnx_op_mapping[node_entry['op'].name]()
        node_entry['output_names'] = [self._tuple_to_name([idx, 0, 0])]
        node_entry['input_names'] = []
        for input_idx_tuple in node_entry['inputs']:
            if self._node_list[input_idx_tuple[0]]['name']:
                node_entry['input_names'].append(self._node_list[input_idx_tuple[0]]['name'])
            else:
                node_entry['input_names'].append(self._tuple_to_name(input_idx_tuple))

        converter.convert(node_entry, self._mc, self._node_list)

    def _add_params(self, node_entry, idx):
        """Add param value to initializer and name to inputs"""
        param_name = node_entry['name']
        assert param_name in self._params, "The parameter {0} is not present" \
                                           "in params dict provided.".format(param_name)
        value = self._params[param_name]
        numpy_array = value.asnumpy()
        tensor = numpy_helper.from_array(numpy_array, param_name)
        self._mc.add_initializers([tensor])
        dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy_array.dtype]
        input = onnx.helper.make_tensor_value_info(param_name,
                                                   dtype,
                                                   shape=numpy_array.shape)
        self._mc.add_inputs([input])

    def _add_constant_input(self, node_entry, idx):
        """Create named input for constant and add it to container inputs.
        If input is a parameter then add to param
        """
        node = node_entry['node']
        if not node_entry['name']:
            node_entry['name'] = self._tuple_to_name([idx, 0, 0])
        param_name = node_entry['name']
        self._params[param_name] = node.data
        self._add_params(node_entry, idx)

    def _add_input(self, node_entry, idx):
        """Add input node to container inputs. If input is a parameter then add to param"""
        if node_entry['name'] in self._params:
            self._add_params(node_entry, idx)
        else:
            type = node_entry['types'][0]
            dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(type.dtype)]
            input = onnx.helper.make_tensor_value_info(node_entry['name'],
                                                       dtype,
                                                       shape=type.concrete_shape)
            self._mc.add_inputs([input])

    def _add_output(self, node_entry, idx):
        """Add output node to container outputs."""

        type = node_entry['types'][0]
        dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(type.dtype)]
        output = onnx.helper.make_tensor_value_info(self._tuple_to_name([idx, 0, 0]),
                                                    dtype,
                                                    shape=type.concrete_shape)
        self._mc.add_outputs([output])


def to_onnx(relay_module, params, name, opset_version=11, path=None):
    """Convert a Relay Function Module into an equivalent ONNX and serialize it to the path

    Parameters
    ----------
    relay_module : tvm.relay.Module
        The relay module object

    params : dict
        dict of the parameter names and NDarray values

    path : str
        The path where ONNX model will be saved

    Returns
    -------
    inferred_model : tvm.relay.Module
        The relay module

    """

    if opset_version not in ONNX_OPSET_VERSONS_SUPPORTED:
        raise NotImplementedError("Currently only opset version 11 is supported.")

    if opset_version > defs.onnx_opset_version():
        raise Exception("The ONNX package installed of version {} does not support the opset "
                        "version {}. Upgrade the ONNX package to latest version.".format(
                            get_onnx_version(), opset_version))

    node_list = []  # ONNX needs a topologically sorted list of nodes
    node_dict = {}
    _expr2graph_impl(relay_module["main"], [], node_dict, node_list)
    converter = RelayToONNXConverter(name, node_list, params, opset_version)
    onnx_model = converter.convert_to_onnx()

    if path:
        onnx.save(onnx_model, path)
    return onnx_model
