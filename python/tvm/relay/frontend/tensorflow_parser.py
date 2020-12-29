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
    model_dir : tensorflow frozen pb file or a directory that contains saved
    model or checkpoints.

    outputs : List of output tensor names (Optional)
        Optional output node names. This will be protected for saved model
        when we do remove training nodes.

    Examples
    --------
    .. code-block:: python

        parser = TFParser(model_dir)
        graphdef = parser.parse()
    """

    def __init__(self, model_dir, outputs=None):
        from tensorflow.core.framework import graph_pb2

        self._tmp_dir = utils.tempdir()
        self._model_dir = model_dir
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
        with open(self._model_dir, "rb") as f:
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
        tag_sets = get_saved_model_tag_sets(self._model_dir)
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
            meta_graph_def = tf.saved_model.loader.load(sess, tags, self._model_dir)
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

        saved_model_dir = self._model_dir
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

    def parse(self):
        """
        Parse tensorflow models: checkpoints, saved models, and single frozen pb file.

        Returns
        -------
        GraphDef of the passed model
        """

        graph = None

        if os.path.isdir(self._model_dir):
            ckpt = os.path.join(self._model_dir, "checkpoint")
            if not os.path.isfile(ckpt):
                if not os.path.isdir(os.path.join(self._model_dir, "variables")):
                    raise RuntimeError("InputConfiguration: Invalid model path.")
                graph = self._load_saved_model()
            else:
                graph = self._load_ckpt()
        elif os.path.isfile(self._model_dir):
            # Only .pb or .pbtxt is a valid suffix name.
            if self._model_dir.endswith(".pb") or self._model_dir.endswith(".pbtxt"):
                cur_dir = os.path.dirname(self._model_dir)
            else:
                raise RuntimeError("InputConfiguration: Invalid model format.")

            # It is a saved model if `variables` directory is present at the
            # same directory with the pb or pbtxt file.
            if os.path.isdir(os.path.join(cur_dir, "variables")):
                self._model_dir = cur_dir
                graph = self._load_saved_model()
            else:
                graph = self._load_pb_file()
        else:
            raise RuntimeError("InputConfiguration: Unrecognized model " "file or path.")

        self._set_graph(graph)
        return graph
