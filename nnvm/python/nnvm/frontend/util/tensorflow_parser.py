"""TF: Tensorflow parser"""
from __future__ import absolute_import as _abs
from __future__ import print_function
import os

try:
    from tensorflow.core.framework import graph_pb2
except ImportError as e:
    from nnvm.frontend.protobuf import graph_pb2


try:
    from tempfile import TemporaryDirectory
except ImportError:
    import tempfile
    import shutil

    class TemporaryDirectory(object):
        def __enter__(self):
            self.name = tempfile.mkdtemp()
            return self.name

        def __exit__(self, exc, value, tb):
            shutil.rmtree(self.name)


class TFParser(object):
    """A Wrapper to handle tensorflow models parsing
       Works w/o installing tensorflow,
       Protocol Buffer is needed
    ```
    parser = TfParser(model_dir)
    graph = parser.parse()
    ```
    Parameters
    ----------
    model_dir : tensorflow frozen pb file or a directory that contains saved
    model or checkpoints.
    """

    def __init__(self, model_dir):
        self._tmp_dir = TemporaryDirectory()
        self._model_dir = model_dir
        self._graph = graph_pb2.GraphDef()

    def _set_graph(self, graph):
        """Set Graph"""
        self._graph = graph

    def _get_graph(self):
        """Get Graph"""
        return self._graph

    def _output_graph(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)
        for node in self._get_graph().node:
            logging.info("Name: {}".format(node.name))
            logging.info("\top: {}".format(node.op))
            for input in node.input:
                logging.info("\t\tinput: {}".format(input))
            logging.info("\t\tdevice: {}".format(node.device))
            logging.info("\t\tAttrValue: ")
            for key in node.attr.keys():
                logging.info("\t\t\tkey: {} => value: {}"
                             .format(key, node.attr[key]))
            logging.info(node.attr['shape'].shape)

    def _load_pb_file(self):
        """Load single pb file"""
        graph = self._get_graph()
        with open(self._model_dir, "rb") as f:
            graph.ParseFromString(f.read())
        return graph

    def _get_output_names(self, model_path):
        """Return the concatenated output names"""
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "InputConfiguration: Unable to import tensorflow which is "
                "required to restore from saved model. {}".format(e))

        with tf.Session() as sess:
            meta_graph_def = tf.saved_model.loader.load(sess,
                                                        [tf.saved_model.tag_constants.SERVING],
                                                        model_path)
            output_names = set()
            for k in meta_graph_def.signature_def.keys():
                outputs_tensor_info = meta_graph_def.signature_def[k].outputs
                for output_tensor in outputs_tensor_info.values():
                    output_names.add(output_tensor.name)
            output_names = [i.replace(":0", "") for i in output_names]
            return ",".join(output_names)

    def _load_saved_model(self):
        """Load the tensorflow saved model."""
        try:
            import tensorflow as tf
            from tensorflow.python.tools import freeze_graph
            from tensorflow.python.framework import ops
            from tensorflow.python.framework import graph_util
        except ImportError as e:
            raise ImportError(
                "InputConfiguration: Unable to import tensorflow which is "
                "required to restore from saved model. {}".format(e))

        saved_model_dir = self._model_dir
        output_graph_filename = os.path.join(self._tmp_dir.name, "neo_frozen_model.pb")
        input_saved_model_dir = saved_model_dir
        output_node_names = self._get_output_names(self._model_dir)

        input_binary = False
        input_saver_def_path = False
        restore_op_name = None
        filename_tensor_name = None
        clear_devices = True
        input_meta_graph = False
        checkpoint_path = None
        input_graph_filename = None
        saved_model_tags = tf.saved_model.tag_constants.SERVING

        freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                                  input_binary, checkpoint_path, output_node_names,
                                  restore_op_name, filename_tensor_name,
                                  output_graph_filename, clear_devices, "", "", "",
                                  input_meta_graph, input_saved_model_dir,
                                  saved_model_tags)

        with ops.Graph().as_default():
            output_graph_def = graph_pb2.GraphDef()
            with open(output_graph_filename, "rb") as f:
                output_graph_def.ParseFromString(f.read())
            output_graph_def = graph_util.remove_training_nodes(output_graph_def)
            return output_graph_def

    def _load_ckpt(self):
        """TODO: Load checkpoint model."""
        raise RuntimeError("InputConfiguration: Loading tf checkpoint model is "
                           "not supported yet.")

    def parse(self):
        """Parse tensorflow models: checkpoints, saved models, and single pb
        file.
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
            if self._model_dir.endswith(".pb") or \
               self._model_dir.endswith(".pbtxt"):
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
            raise RuntimeError("InputConfiguration: Unrecognized model "
                               "file or path.")

        self._set_graph(graph)
        return graph
