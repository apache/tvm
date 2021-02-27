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
"""TF: Tensorflow parser"""
# pylint: disable=import-outside-toplevel, assignment-from-no-return

import os
from tvm.contrib import utils


class TFParser(object):
    """
    A Wrapper to handle tensorflow models parsing, TensorFlow is needed

    Parameters
    ----------
    tf_input : tensorflow model inpu. Could be any one of the following.
        file: Protobuf file
        directory: saved_model or check point.
        GraphDef: tensorflow.core.framework.graph_pb2.GraphDef
        Concrete Function: tensorflow.python.eager.function.ConcreteFunction

    outputs : List of output tensor names (Optional)
        Optional output node names. This will be protected for saved model
        when we do remove training nodes.

    Examples
    --------
    .. code-block:: python

        parser = TFParser(tf_input)
        graphdef = parser.parse()
    """

    def __init__(self, tf_input, outputs=None):
        from tensorflow.core.framework import graph_pb2

        self._tmp_dir = utils.tempdir()
        self._tf_input = tf_input
        self._graph = graph_pb2.GraphDef()
        self._outputs = outputs or []

    def _set_graph(self, graph):
        """Set Graph"""
        self._graph = graph

    def _get_graph(self):
        """Get Graph"""
        return self._graph

    def _load_pb_file(self):
        """Load single pb file"""
        graph = self._get_graph()
        with open(self._tf_input, "rb") as f:
            graph.ParseFromString(f.read())
        return graph

    def _get_tag_set(self):
        """Return the tag set of saved model, multiple metagraphs are not supported"""
        try:
            from tensorflow.contrib.saved_model.python.saved_model.reader import (
                get_saved_model_tag_sets,
            )
        except ImportError:
            try:
                from tensorflow.python.tools.saved_model_utils import get_saved_model_tag_sets
            except ImportError:
                raise ImportError(
                    "InputConfiguration: Unable to import get_saved_model_tag_sets which is "
                    "required to get tag set from saved model."
                )
        tag_sets = get_saved_model_tag_sets(self._tf_input)
        return tag_sets[0]

    def _get_output_names(self):
        """Return the concatenated output names"""
        try:
            import tensorflow.compat.v1 as tf
        except ImportError:
            raise ImportError(
                "InputConfiguration: Unable to import tensorflow which is "
                "required to restore from saved model."
            )
        tags = self._get_tag_set()
        output_names = set()
        with tf.Session() as sess:
            meta_graph_def = tf.saved_model.loader.load(sess, tags, self._tf_input)
            for sig_def in meta_graph_def.signature_def.values():
                for output_tensor in sig_def.outputs.values():
                    output_names.add(output_tensor.name.replace(":0", ""))
        tf.reset_default_graph()
        return ",".join(output_names)

    def _load_saved_model(self):
        """Load the tensorflow saved model."""
        try:
            from tensorflow.python.tools import freeze_graph
            from tensorflow.python.framework import ops
            from tensorflow.python.framework import graph_util
            from tensorflow.core.framework import graph_pb2
        except ImportError:
            raise ImportError(
                "InputConfiguration: Unable to import tensorflow which is "
                "required to restore from saved model."
            )

        saved_model_dir = self._tf_input
        output_graph_filename = self._tmp_dir.relpath("tf_frozen_model.pb")
        input_saved_model_dir = saved_model_dir
        output_node_names = self._get_output_names()

        input_binary = False
        input_saver_def_path = False
        restore_op_name = None
        filename_tensor_name = None
        clear_devices = True
        input_meta_graph = False
        checkpoint_path = None
        input_graph_filename = None
        saved_model_tags = ",".join(self._get_tag_set())

        freeze_graph.freeze_graph(
            input_graph_filename,
            input_saver_def_path,
            input_binary,
            checkpoint_path,
            output_node_names,
            restore_op_name,
            filename_tensor_name,
            output_graph_filename,
            clear_devices,
            "",
            "",
            "",
            input_meta_graph,
            input_saved_model_dir,
            saved_model_tags,
        )

        with ops.Graph().as_default():  # pylint: disable=not-context-manager
            output_graph_def = graph_pb2.GraphDef()
            with open(output_graph_filename, "rb") as f:
                output_graph_def.ParseFromString(f.read())
            output_graph_def = graph_util.remove_training_nodes(
                output_graph_def, protected_nodes=self._outputs
            )
            return output_graph_def

    def _load_ckpt(self):
        """TODO: Load checkpoint model."""
        raise RuntimeError(
            "InputConfiguration: Loading tf checkpoint model is " "not supported yet."
        )

    # Ref. taken from Tensorflow JS
    def _build_signature_def(self, frozen_graph, input_nodes, output_nodes):
        try:
            from tensorflow.core.protobuf import meta_graph_pb2
        except ImportError as e:
            raise ImportError("Unable to import tensorflow which is required {}".format(e))
        signature = meta_graph_pb2.SignatureDef()
        for input_tensor in input_nodes:
            op_name = input_tensor.name.split(":")[0]
            # The graph freezing may turn the original inputs into constants, or remove
            # them from the graph, so we need to ignore those.
            try:
                op = frozen_graph.get_operation_by_name(op_name)
                if op.type != "Const":
                    signature.inputs[input_tensor.name].name = input_tensor.name
                    signature.inputs[input_tensor.name].dtype = input_tensor.dtype.as_datatype_enum
                    signature.inputs[input_tensor.name].tensor_shape.CopyFrom(
                        input_tensor.shape.as_proto()
                    )
            except KeyError:
                # The original input was removed when the graph was frozen.
                continue
        for output_tensor in output_nodes:
            if hasattr(output_tensor, "name"):
                signature.outputs[output_tensor.name].name = output_tensor.name
                signature.outputs[output_tensor.name].dtype = output_tensor.dtype.as_datatype_enum
                signature.outputs[output_tensor.name].tensor_shape.CopyFrom(
                    output_tensor.shape.as_proto()
                )
            else:  # just the tensor name string array
                signature.outputs[output_tensor].name = output_tensor
        return signature


    def _run_grappler(self, config, graph_def, graph, signature_def):
        try:
            from tensorflow.python.grappler import tf_optimizer
            from tensorflow.python.training.saver import export_meta_graph
        except ImportError as e:
            raise ImportError("Unable to import tensorflow which is required {}".format(e))
        meta_graph = export_meta_graph(graph_def=graph_def, graph=graph)
        meta_graph.signature_def["not_used_key"].CopyFrom(signature_def)
        return tf_optimizer.OptimizeGraph(config, meta_graph)


    def parse(self):
        """
        Parse tensorflow models: checkpoints, saved models, and single frozen pb file.

        Returns
        -------
        GraphDef of the passed model
        """

        graph = None
        from tensorflow.python.eager.function import ConcreteFunction
        from tensorflow.core.framework.graph_pb2 import GraphDef

        if isinstance(self._tf_input, ConcreteFunction):
            try:
                from tensorflow.python.framework import convert_to_constants
                from tensorflow.core.protobuf import config_pb2
            except ImportError as e:
                raise ImportError("Unable to import tensorflow which is required {}".format(e))
            concrete_func = self._tf_input
            graph = convert_to_constants.convert_variables_to_constants_v2(concrete_func).graph
            signature = self._build_signature_def(graph, concrete_func.inputs, concrete_func.outputs)
            graph_def = graph.as_graph_def()

            # Some optimization
            config = config_pb2.ConfigProto()
            rewriter_config = config.graph_options.rewrite_options
            rewriter_config.optimizers[:] = [
                "debug_stripper",
                "arithmetic",
                "dependency",
                "arithmetic",
                "dependency",
            ]
            graph = self._run_grappler(config, graph_def, graph, signature)
        elif isinstance(self._tf_input, GraphDef):
            graph = self._tf_input
        elif os.path.isdir(self._tf_input):
            ckpt = os.path.join(self._tf_input, "checkpoint")
            if not os.path.isfile(ckpt):
                if not os.path.isdir(os.path.join(self._tf_input, "variables")):
                    raise RuntimeError("InputConfiguration: Invalid model path.")
                graph = self._load_saved_model()
            else:
                graph = self._load_ckpt()
        elif os.path.isfile(self._tf_input):
            # Only .pb or .pbtxt is a valid suffix name.
            if self._tf_input.endswith(".pb") or self._tf_input.endswith(".pbtxt"):
                cur_dir = os.path.dirname(self._tf_input)
            else:
                raise RuntimeError("InputConfiguration: Invalid model format.")

            # It is a saved model if `variables` directory is present at the
            # same directory with the pb or pbtxt file.
            if os.path.isdir(os.path.join(cur_dir, "variables")):
                self._tf_input = cur_dir
                graph = self._load_saved_model()
            else:
                graph = self._load_pb_file()
        else:
            raise RuntimeError("InputConfiguration: Unrecognized input " "file or path.")

        self._set_graph(graph)
        return graph
