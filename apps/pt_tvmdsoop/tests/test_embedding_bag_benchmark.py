import tvm
from tvm import relay
import torch
from torch.nn import functional as F
import torch.utils.benchmark as benchmark


class EmbeddingBag(torch.nn.Module):
    def __init__(
        self,
        offsets=None,
        mode="mean",
        padding_idx=None,
        scale_grad_by_freq=False,
        include_last_offset=False,
        per_sample_weights=None,
        sparse=False,
    ):
        super().__init__()
        self.offsets = offsets
        self.mode = mode
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq
        self.include_last_offset = include_last_offset
        self.per_sample_weights = per_sample_weights
        self.sparse = sparse

    def forward(self, inputs, weights):
        return F.embedding_bag(
            inputs,
            weights,
            offsets=self.offsets,
            mode=self.mode,
            padding_idx=self.padding_idx,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
            include_last_offset=self.include_last_offset,
            per_sample_weights=self.per_sample_weights,
        )


def timeit(stmt, setup, global_dict, description, sub_label):
    results.append(
        benchmark.Timer(
            stmt=stmt,
            setup=setup,
            globals=global_dict,
            sub_label=sub_label,
            description=description,
        ).blocked_autorange()
    )


def get_exe(func, baseline_input):
    torch_mod = torch.jit.trace(func, baseline_input)
    input_names = [f"input{idx}" for idx, _ in enumerate(baseline_input)]
    input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    mod, params = relay.frontend.from_pytorch(torch_mod, input_shapes, None)
    compiled_input = dict(zip(input_names, [inp.clone().cpu().numpy() for inp in baseline_input]))
    exe = relay.create_executor(
        kind="graph", mod=mod, params=params, device=tvm.device("cuda", 0), target="cuda"
    ).evaluate()
    return exe, compiled_input, torch_mod


weight = torch.rand(10, 3).cuda()
inp = torch.tensor([[1, 1, 4], [5, 9, 4], [4, 2, 5]]).cuda()
offset = torch.tensor([0, 4, 6]).cuda()
per_sample_weights = torch.tensor([[0.3, 0.4, 0.4], [0.1, 2.2, 3.1], [0.2, 1.1, 3.2]]).cuda()

results = []

# test 1
baseline_input = (inp, weight)
func = EmbeddingBag()
exe, compiled_input, torch_mod = get_exe(func, baseline_input)
for i in range(5):
    label = f"2d input: test {i}"
    timeit(
        "exe(**compiled_input)",
        "from __main__ import exe",
        {"compiled_input": compiled_input},
        "relay",
        label,
    )
    timeit(
        "torch_mod(inp, weight)",
        "from __main__ import torch_mod",
        {"inp": inp, "weight": weight},
        "pytorch",
        label,
    )

# test 2
baseline_input = (inp.reshape(-1), weight)
func = EmbeddingBag(offsets=offset)
exe, compiled_input, torch_mod = get_exe(func, baseline_input)
for i in range(5):
    label = f"1d input: test {i}"
    timeit(
        "exe(**compiled_input)",
        "from __main__ import exe",
        {"compiled_input": compiled_input},
        "relay",
        label,
    )
    timeit(
        "torch_mod(inp, weight)",
        "from __main__ import torch_mod",
        {"inp": inp.reshape(-1), "weight": weight},
        "pytorch",
        label,
    )

# test 3
baseline_input = (inp, weight)
func = EmbeddingBag(per_sample_weights=per_sample_weights, mode="sum")
exe, compiled_input, torch_mod = get_exe(func, baseline_input)
for i in range(5):
    label = f"per_sample_weights: test {i}"
    timeit(
        "exe(**compiled_input)",
        "from __main__ import exe",
        {"compiled_input": compiled_input},
        "relay",
        label,
    )
    timeit(
        "torch_mod(inp, weight)",
        "from __main__ import torch_mod",
        {"inp": inp, "weight": weight},
        "pytorch",
        label,
    )

# test 4
baseline_input = (inp, weight)
func = EmbeddingBag(padding_idx=4)
exe, compiled_input, torch_mod = get_exe(func, baseline_input)
for i in range(5):
    label = f"padding_idx: test {i}"
    timeit(
        "exe(**compiled_input)",
        "from __main__ import exe",
        {"compiled_input": compiled_input},
        "relay",
        label,
    )
    timeit(
        "torch_mod(inp, weight)",
        "from __main__ import torch_mod",
        {"inp": inp, "weight": weight},
        "pytorch",
        label,
    )

# test 5
baseline_input = (inp, weight)
func = EmbeddingBag(per_sample_weights=per_sample_weights, padding_idx=4, mode="sum")
exe, compiled_input, torch_mod = get_exe(func, baseline_input)
for i in range(5):
    label = f"per_sample_weights, padding_idx: test {i}"
    timeit(
        "exe(**compiled_input)",
        "from __main__ import exe",
        {"compiled_input": compiled_input},
        "relay",
        label,
    )
    timeit(
        "torch_mod(inp, weight)",
        "from __main__ import torch_mod",
        {"inp": inp, "weight": weight},
        "pytorch",
        label,
    )

# result
compare = benchmark.Compare(results)
compare.print()
