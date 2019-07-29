import tvm
from tvm import relay
from tvm.relay.testing import resnet
from tvm.relay import transform

def test_change_batch_resnet():
    net, params = resnet.get_workload()
    new_net = transform.ChangeBatch({net["main"].params[0]: 0}, batch_size=123)(net)
    assert new_net["main"].checked_type.ret_type == relay.TensorType((123, 1000))

if __name__ == "__main__":
    test_change_batch_resnet()
