import tvm

@tvm.target.generic_func
def mygeneric(data):
    # default generic function
    return data + 1

@mygeneric.register(["cuda", "gpu"])
def cuda_func(data):
    return data + 2

@mygeneric.register("rocm")
def rocm_func(data):
    return data + 3

@mygeneric.register("cpu")
def rocm_func(data):
    return data + 10


def test_target_dispatch():
    with tvm.target.cuda():
        assert mygeneric(1) == 3

    with tvm.target.rocm():
        assert mygeneric(1) == 4

    with tvm.target.create("cuda"):
        assert mygeneric(1) == 3

    with tvm.target.rasp():
        assert mygeneric(1) == 11

    with tvm.target.create("metal"):
        assert mygeneric(1) == 3

    assert tvm.target.current_target() == None

def test_target_string_parse():
    target = tvm.target.create("cuda -libs=cublas,cudnn")

    assert target.target_name == "cuda"
    assert target.options == ['-libs=cublas,cudnn']
    assert target.keys == ['cuda', 'gpu']
    assert target.libs == ['cublas', 'cudnn']
    assert str(target) == str(tvm.target.cuda("-libs=cublas,cudnn"))

if __name__ == "__main__":
    test_target_dispatch()
    test_target_string_parse()
