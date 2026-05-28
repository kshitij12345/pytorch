"""
Reproducer 2: Combo kernel reduction store with XBLOCK=1 (too small for TMA)

Error: Descriptor block shape must have at least 16 bytes in the last
dimension, but got 1 * 4 = 4 bytes

Root cause: In a combo reduction kernel, a sub-kernel has XBLOCK=1 (from
the output reduction) and element_size=4 (float32). 1 * 4 = 4 < 16 byte
minimum for TMA. The existing `no_x_dim` guard doesn't trigger because
the combo kernel sets x_dim but with block=1.

Original test: test/inductor/test_combo_kernels.py::ComboKernelTests::test_combo_kernel_per_config_subkernel_red

Run: python agent_space/tma_repros/repro_02_combo_kernel_small_block.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMAComboKernelSmallBlock(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": False,
        }
    )
    def test_combo_kernel_reduction_xblock_1(self):
        def fn(a, b):
            r1 = a.sum(dim=(0, 2))
            r2 = b.sum(dim=(0, 2))
            return r1, r2

        inps = [
            torch.randn(32, 64, 128, device="cuda"),
            torch.randn(32, 64, 128, device="cuda"),
        ]
        expected = fn(*inps)
        compiled_fn = torch.compile(fn)
        actual = compiled_fn(*inps)
        torch.testing.assert_close(expected, actual, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
