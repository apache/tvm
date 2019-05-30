import tvm
from tvm import relay
import tvm.relay.testing
from tvm.relay.visualize import visualize

def test_visualize():
    resnet, params = relay.testing.resnet.get_workload()
    visualize(resnet)


if __name__ == "__main__":
    test_visualize()
