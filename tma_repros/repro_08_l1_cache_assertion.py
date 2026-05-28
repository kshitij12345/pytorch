"""
Reproducer 8: TMA changes load codegen breaking L1 cache eviction assertions

Error: AssertionError: [] is not true
(No tl.load lines found in generated code, because TMA replaced them all
with tensor descriptors, but the test expects traditional tl.load calls)

Root cause: When TMA is globally enabled, pointwise kernels that previously
used `tl.load` with eviction policies now use `tl.make_tensor_descriptor`
loads instead. Tests that inspect the generated code for `tl.load` patterns
(like cache eviction policy tests) fail because the loads have been replaced.
This is more of a test/code interaction issue - the TMA path doesn't support
explicit eviction policy control.

Original test: test/inductor/test_torchinductor.py::TritonCodeGenTests::test_skip_l1_cache_buf_read_counts_guard

Run: python agent_space/tma_repros/repro_08_l1_cache_assertion.py
"""

import torch
import torch.nn as nn
import torch._inductor.config as inductor_config
from torch._inductor.codegen import simd_kernel_features
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._inductor.utils import run_and_get_triton_code


class TestTMAL1CacheAssertion(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_l1_cache_tl_load_replaced_by_tma(self):
        class M(nn.Module):
            def __init__(self, n: int):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(n))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * self.weight

        N = 512
        m = torch.compile(M(N).to(device="cuda"))
        x = torch.randn(N, device="cuda")

        orig = simd_kernel_features.SIMDKernelFeatures.buffer_read_counts

        def fake_buffer_read_counts(self_inner):
            return {}

        simd_kernel_features.SIMDKernelFeatures.buffer_read_counts = (
            fake_buffer_read_counts
        )
        try:
            code = run_and_get_triton_code(m, x)
            lines = [line for line in code.split("\n") if "tl.load" in line]
            self.assertTrue(lines, "Expected tl.load calls but TMA replaced them all")
        finally:
            simd_kernel_features.SIMDKernelFeatures.buffer_read_counts = orig


if __name__ == "__main__":
    run_tests()
