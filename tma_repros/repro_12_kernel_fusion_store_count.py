"""
Reproducer 12: TMA changes kernel fusion decisions (kernel count mismatch)

Error: AssertionError: Scalars are not equal!
  Expected 1 but got 0.
  (code.count("tl.store") returns wrong count because TMA replaced stores)

Root cause: When TMA tensor descriptors are enabled, the generated Triton
code uses `tl.make_tensor_descriptor(...).store(...)` instead of `tl.store(...)`.
Tests that count `tl.store` occurrences to verify fusion behavior get 0 instead
of the expected count because the store pattern has changed.

Original test: test/inductor/test_torchinductor.py::TritonCodeGenTests::test_not_materialize_pointwise_reduction

Run: python agent_space/tma_repros/repro_12_kernel_fusion_store_count.py
"""

import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMAKernelFusionStoreCount(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_not_materialize_pointwise_reduction(self):
        def fn(a, b):
            return (a - b).sum(dim=-1).amax(dim=-1)

        N = 16
        K = 7
        fn_opt = torch.compile(fn, backend="inductor")
        inps = [
            torch.randn(N, 1, K, device="cuda"),
            torch.randn(1, N, K, device="cuda"),
        ]
        code = run_and_get_triton_code(fn_opt, *inps)
        # With TMA, stores become descriptor.store() not tl.store()
        # The original test asserts count("tl.store") == 1
        store_count = code.count("tl.store")
        self.assertEqual(
            store_count,
            1,
            f"Expected 1 tl.store but got {store_count}. "
            f"TMA replaces tl.store with descriptor.store(), breaking the assertion.",
        )


if __name__ == "__main__":
    run_tests()
