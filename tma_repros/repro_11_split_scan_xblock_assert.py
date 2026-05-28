"""
Reproducer 11: Split scan XBLOCK assertion failure

Error: AssertionError: assert tconfig.kwargs["XBLOCK"] == 1

Root cause: The split_scan heuristic decorator asserts that XBLOCK must be 1
for split scan kernels (since they process one x-element at a time with a
large reduction). When TMA is enabled, the autotuning configs may include
XBLOCK values > 1 because the TMA path generates different block size
configurations. The assertion fires before the kernel can compile.

Original test: test/inductor/test_torchinductor.py::GPUTests::test_cumprod_backward_split_scan_reduction_fusion_cuda

Run: python agent_space/tma_repros/repro_11_split_scan_xblock_assert.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMASplitScanXblockAssert(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_cumprod_backward_split_scan(self):
        seq_len = 8193
        channels = 64

        def fn(x, gamma):
            decay = gamma.view(1, 1, channels).expand_as(x)
            retention = torch.cumprod(decay, dim=1)
            return (x * retention).sum()

        x = torch.randn(
            2, seq_len, channels, device="cuda", requires_grad=True
        )
        gamma = torch.full(
            (channels,), 0.999, device="cuda"
        ).requires_grad_()

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        y = compiled_fn(x, gamma)
        y.backward()


if __name__ == "__main__":
    run_tests()
