import numpy as np
import nnvm.compiler

def test_save_load():
    x = np.random.uniform(size=(10, 2)).astype("float32")
    y = np.random.uniform(size=(1, 2, 3)).astype("float32")
    x[:] = 1
    y[:] = 1
    params = {"x": x, "y": y}
    param_bytes = nnvm.compiler.save_param_dict(params)
    assert isinstance(param_bytes, bytearray)
    param2 = nnvm.compiler.load_param_dict(param_bytes)
    assert len(param2) == 2
    np.testing.assert_equal(param2["x"].asnumpy(), x)
    np.testing.assert_equal(param2["y"].asnumpy(), y)


if __name__ == "__main__":
    test_save_load()
