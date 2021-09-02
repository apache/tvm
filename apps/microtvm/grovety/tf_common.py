import tvm.relay as relay
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


def get_values(size0, type, size1=1, type2=None):
    """Generate two tensors."""   
    if not type2:
        type2 = type
    range_ = 2
    in_data = np.random.randint(-range_, range_ + 1, size0).astype(type)
    b_data = np.random.randint(-range_, range_ + 1, size1).astype(type2)
    return in_data, b_data


def save_pd_file(file, graph, output_node_names):
    """Write Frozen Graph file to disk."""
    input_graph_def = graph.as_graph_def(add_shapes=True)

    output_graph_def = tf.compat.v1.graph_util.extract_sub_graph(input_graph_def, output_node_names)

    pb_filepath = file
    with tf.io.gfile.GFile(pb_filepath, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.io.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def run_tf(in_data, pb_file, input_node, output_node, outs=None):
    """Run on tensorflow."""
    # launch the default graph.
    _in = input_node + ':0'

    if outs:
        _outs = []
        for each in outs:
            _outs.append(each + ':0')
        _out = _outs
        output_node_names = outs
    else:
        _out = output_node + ':0'
        output_node_names = [output_node]

    config = None

    with tf.compat.v1.Session(config=config) as sess:
        tf_val = sess.run(_out, {_in: in_data})

        # freeze graph
        graph = tf.compat.v1.get_default_graph()
        save_pd_file(pb_file, graph, output_node_names)

    return tf_val


def open_conv2d_model(pb_file, input_tensor, input_shape, layout, outs=None):    
    import tvm.relay.testing.tf as tf_testing

    frozen_graph_def = get_frozen_graph(pb_file)
    graph_def = tf_testing.ProcessGraphDefParam(frozen_graph_def)

    relay_mod, params = relay.frontend.from_tensorflow(graph_def, shape={input_tensor: input_shape}, layout=layout, outputs=outs)

    return (relay_mod, params)
